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

#include "CodeGenerator.h"
#include "Execute.h"

#include <future>
#include <memory>

llvm::Value* CodeGenerator::codegen(const hdk::ir::InValues* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto in_arg = expr->arg();
  if (is_unnest(in_arg)) {
    throw std::runtime_error("IN not supported for unnested expressions");
  }
  auto expr_type = expr->type();
  CHECK(expr_type->isBoolean());
  const auto lhs_lvs = codegen(in_arg, true, co);
  llvm::Value* result{nullptr};
  if (expr_type->nullable()) {
    result = cgen_state_->llInt(int8_t(0));
  } else {
    result = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_),
                                    false);
  }
  CHECK(result);
  if (co.hoist_literals) {  // TODO(alex): remove this constraint
    auto in_vals_bitmap = createInValuesBitmap(expr, co);
    if (in_vals_bitmap) {
      if (in_vals_bitmap->isEmpty()) {
        return in_vals_bitmap->hasNull()
                   ? cgen_state_->inlineIntNull(expr_type->ctx().boolean())
                   : result;
      }
      CHECK_EQ(size_t(1), lhs_lvs.size());
      return cgen_state_->addInValuesBitmap(in_vals_bitmap)
          ->codegen(lhs_lvs.front(), executor(), co.codegen_traits_desc);
    }
  }
  if (!expr_type->nullable()) {
    for (auto in_val : expr->valueList()) {
      result =
          cgen_state_->ir_builder_.CreateOr(result,
                                            toBool(codegenCmp(hdk::ir::OpType::kEq,
                                                              hdk::ir::Qualifier::kOne,
                                                              lhs_lvs,
                                                              in_arg->type(),
                                                              in_val.get(),
                                                              co)));
    }
  } else {
    for (auto in_val : expr->valueList()) {
      const auto crt = codegenCmp(hdk::ir::OpType::kEq,
                                  hdk::ir::Qualifier::kOne,
                                  lhs_lvs,
                                  in_arg->type(),
                                  in_val.get(),
                                  co);
      result = cgen_state_->emitCall(
          "logical_or", {result, crt, cgen_state_->inlineIntNull(expr_type)});
    }
  }
  return result;
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::InIntegerSet* in_integer_set,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto in_arg = in_integer_set->arg();
  if (is_unnest(in_arg)) {
    throw std::runtime_error("IN not supported for unnested expressions");
  }
  auto type = in_integer_set->arg()->type();
  const auto needle_null_val = inline_int_null_value(type);
  if (!co.hoist_literals) {
    // We never run without literal hoisting in real world scenarios, this avoids a crash
    // when testing.
    throw std::runtime_error(
        "IN subquery with many right-hand side values not supported when literal "
        "hoisting is disabled");
  }
  auto in_vals_bitmap = std::make_unique<InValuesBitmap>(
      in_integer_set->valueList(),
      needle_null_val,
      co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                : Data_Namespace::CPU_LEVEL,
      executor()->deviceCount(co.device_type),
      executor()->getBufferProvider());
  auto in_integer_set_type = in_integer_set->type();
  CHECK(in_integer_set_type->isBoolean());
  const auto lhs_lvs = codegen(in_arg, true, co);
  llvm::Value* result{nullptr};
  if (in_integer_set_type->nullable()) {
    result = cgen_state_->llInt(int8_t(0));
  } else {
    result = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_),
                                    false);
  }
  CHECK(result);
  CHECK_EQ(size_t(1), lhs_lvs.size());
  return cgen_state_->addInValuesBitmap(in_vals_bitmap)
      ->codegen(lhs_lvs.front(), executor(), co.codegen_traits_desc);
}

std::unique_ptr<InValuesBitmap> CodeGenerator::createInValuesBitmap(
    const hdk::ir::InValues* in_values,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto& value_list = in_values->valueList();
  const auto val_count = value_list.size();
  auto type = in_values->arg()->type();
  if (!(type->isInteger() || type->isExtDictionary())) {
    return nullptr;
  }
  const auto sdp = type->isExtDictionary()
                       ? executor()->getStringDictionaryProxy(
                             type->as<hdk::ir::ExtDictionaryType>()->dictId(),
                             executor()->getRowSetMemoryOwner(),
                             true)
                       : nullptr;
  if (val_count > 3) {
    using ListIterator = decltype(value_list.begin());
    std::vector<int64_t> values;
    const auto needle_null_val = inline_int_null_value(type);
    const int worker_count = val_count > 10000 ? cpu_threads() : int(1);
    std::vector<std::vector<int64_t>> values_set(worker_count, std::vector<int64_t>());
    std::vector<std::future<bool>> worker_threads;
    auto start_it = value_list.begin();
    for (size_t i = 0,
                start_val = 0,
                stride = (val_count + worker_count - 1) / worker_count;
         i < val_count && start_val < val_count;
         ++i, start_val += stride, std::advance(start_it, stride)) {
      auto end_it = start_it;
      std::advance(end_it, std::min(stride, val_count - start_val));
      const auto do_work = [&](std::vector<int64_t>& out_vals,
                               const ListIterator start,
                               const ListIterator end) -> bool {
        for (auto val_it = start; val_it != end; ++val_it) {
          const auto& in_val = *val_it;
          const auto in_val_const =
              dynamic_cast<const hdk::ir::Constant*>(extract_cast_arg(in_val.get()));
          if (!in_val_const) {
            return false;
          }
          auto in_val_type = in_val->type();
          CHECK(in_val_type->equal(type) || in_val_type->withNullable(true)->equal(type));
          if (type->isExtDictionary()) {
            CHECK(sdp);
            const auto string_id =
                in_val_const->isNull()
                    ? needle_null_val
                    : sdp->getIdOfString(*in_val_const->value().stringval);
            if (string_id != StringDictionary::INVALID_STR_ID) {
              out_vals.push_back(string_id);
            }
          } else {
            out_vals.push_back(CodeGenerator::codegenIntConst(in_val_const, cgen_state_)
                                   ->getSExtValue());
          }
        }
        return true;
      };
      if (worker_count > 1) {
        worker_threads.push_back(std::async(
            std::launch::async, do_work, std::ref(values_set[i]), start_it, end_it));
      } else {
        do_work(std::ref(values), start_it, end_it);
      }
    }
    bool success = true;
    for (auto& worker : worker_threads) {
      success &= worker.get();
    }
    if (!success) {
      return nullptr;
    }
    if (worker_count > 1) {
      size_t total_val_count = 0;
      for (auto& vals : values_set) {
        total_val_count += vals.size();
      }
      values.reserve(total_val_count);
      for (auto& vals : values_set) {
        values.insert(values.end(), vals.begin(), vals.end());
      }
    }
    try {
      return std::make_unique<InValuesBitmap>(values,
                                              needle_null_val,
                                              co.device_type == ExecutorDeviceType::GPU
                                                  ? Data_Namespace::GPU_LEVEL
                                                  : Data_Namespace::CPU_LEVEL,
                                              executor()->deviceCount(co.device_type),
                                              executor()->getBufferProvider());
    } catch (...) {
      return nullptr;
    }
  }
  return nullptr;
}
