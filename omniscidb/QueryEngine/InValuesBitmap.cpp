/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "InValuesBitmap.h"
#include "CodeGenerator.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "Logger/Logger.h"
#include "RuntimeFunctions.h"
#include "Shared/checked_alloc.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <limits>

using checked_int64_t = boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<64,
                                           64,
                                           boost::multiprecision::signed_magnitude,
                                           boost::multiprecision::checked,
                                           void>>;

InValuesBitmap::InValuesBitmap(const std::vector<int64_t>& values,
                               const int64_t null_val,
                               const Data_Namespace::MemoryLevel memory_level,
                               const int device_count,
                               BufferProvider* buffer_provider)
    : rhs_has_null_(false)
    , null_val_(null_val)
    , memory_level_(memory_level)
    , device_count_(device_count)
    , buffer_provider_(buffer_provider) {
#if defined(HAVE_CUDA) || defined(HAVE_L0)
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  if (values.empty()) {
    return;
  }
  min_val_ = std::numeric_limits<int64_t>::max();
  max_val_ = std::numeric_limits<int64_t>::min();
  for (const auto value : values) {
    if (value == null_val) {
      rhs_has_null_ = true;
      continue;
    }
    if (value < min_val_) {
      min_val_ = value;
    }
    if (value > max_val_) {
      max_val_ = value;
    }
  }
  if (max_val_ < min_val_) {
    CHECK_EQ(std::numeric_limits<int64_t>::max(), min_val_);
    CHECK_EQ(std::numeric_limits<int64_t>::min(), max_val_);
    CHECK(rhs_has_null_);
    return;
  }
  const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000LL};
  const auto bitmap_sz_bits =
      static_cast<int64_t>(checked_int64_t(max_val_) - min_val_ + 1);
  if (bitmap_sz_bits > MAX_BITMAP_BITS) {
    throw FailedToCreateBitmap();
  }
  const auto bitmap_sz_bytes = bitmap_bits_to_bytes(bitmap_sz_bits);
  auto cpu_bitset = static_cast<int8_t*>(checked_calloc(bitmap_sz_bytes, 1));
  for (const auto value : values) {
    if (value == null_val) {
      continue;
    }
    agg_count_distinct_bitmap(reinterpret_cast<int64_t*>(&cpu_bitset), value, min_val_);
  }
#if defined(HAVE_CUDA) || defined(HAVE_L0)
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      gpu_buffers_.emplace_back(GpuAllocator::allocGpuAbstractBuffer(
          buffer_provider, bitmap_sz_bytes, device_id));
      auto gpu_bitset = gpu_buffers_.back()->getMemoryPtr();
      buffer_provider->copyToDevice(gpu_bitset, cpu_bitset, bitmap_sz_bytes, device_id);
      bitsets_.push_back(gpu_bitset);
    }
    free(cpu_bitset);
  } else {
    bitsets_.push_back(cpu_bitset);
  }
#else
  CHECK_EQ(1, device_count_);
  bitsets_.push_back(cpu_bitset);
#endif  // HAVE_CUDA
}

InValuesBitmap::~InValuesBitmap() {
  if (bitsets_.empty()) {
    return;
  }
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    CHECK_EQ(size_t(1), bitsets_.size());
    free(bitsets_.front());
  } else {
    CHECK(buffer_provider_);
    for (auto& gpu_buffer : gpu_buffers_) {
      buffer_provider_->free(gpu_buffer);
    }
  }
}

llvm::Value* InValuesBitmap::codegen(
    llvm::Value* needle,
    Executor* executor,
    compiler::CodegenTraitsDescriptor codegen_traits_desc) const {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  std::vector<std::shared_ptr<const hdk::ir::Constant>> constants_owned;
  std::vector<const hdk::ir::Constant*> constants;
  for (const auto bitset : bitsets_) {
    const int64_t bitset_handle = reinterpret_cast<int64_t>(bitset);
    const auto bitset_handle_literal = std::dynamic_pointer_cast<const hdk::ir::Constant>(
        Analyzer::analyzeIntValue(bitset_handle));
    CHECK(bitset_handle_literal);
    constants_owned.push_back(bitset_handle_literal);
    constants.push_back(bitset_handle_literal.get());
  }
  const auto needle_i64 = executor->cgen_state_->castToTypeIn(needle, 64);
  const auto null_bool_val = static_cast<int8_t>(inline_null_value<bool>());
  if (bitsets_.empty()) {
    return executor->cgen_state_->emitCall("bit_is_set",
                                           {executor->cgen_state_->llInt(int64_t(0)),
                                            needle_i64,
                                            executor->cgen_state_->llInt(int64_t(0)),
                                            executor->cgen_state_->llInt(int64_t(0)),
                                            executor->cgen_state_->llInt(null_val_),
                                            executor->cgen_state_->llInt(null_bool_val)});
  }
  CodeGenerator code_generator(executor, codegen_traits_desc);
  const auto bitset_handle_lvs =
      code_generator.codegenHoistedConstants(constants, false, 0);
  CHECK_EQ(size_t(1), bitset_handle_lvs.size());
  return executor->cgen_state_->emitCall(
      "bit_is_set",
      {executor->cgen_state_->castToTypeIn(bitset_handle_lvs.front(), 64),
       needle_i64,
       executor->cgen_state_->llInt(min_val_),
       executor->cgen_state_->llInt(max_val_),
       executor->cgen_state_->llInt(null_val_),
       executor->cgen_state_->llInt(null_bool_val)});
}

bool InValuesBitmap::isEmpty() const {
  return bitsets_.empty();
}

bool InValuesBitmap::hasNull() const {
  return rhs_has_null_;
}
