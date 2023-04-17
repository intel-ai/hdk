/*
 * Copyright 2021 OmniSci, Inc.
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

//  project headers
#include "ArrowResultSet.h"
#include "BitmapGenerators.h"
#include "Execute.h"
#include "Shared/ArrowUtil.h"
#include "Shared/DateConverters.h"
#include "Shared/threading.h"
#include "Shared/toString.h"

//  arrow headers
#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/options.h"

// std headers
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>
#include <tuple>

//  TBB headers
#include <tbb/parallel_for.h>

//  OS-specific headers
#ifndef _MSC_VER
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#else
//  IPC shared memory not yet supported on windows
using key_t = size_t;
#define IPC_PRIVATE 0
#endif

#ifdef HAVE_CUDA
#include <arrow/gpu/cuda_api.h>
#include <cuda.h>
#endif  // HAVE_CUDA

//  Definitions
#define ARROW_RECORDBATCH_MAKE arrow::RecordBatch::Make
#define ARROW_CONVERTER_DEBUG true
#define ARROW_LOG(category) \
  VLOG(1) << "[Arrow]"      \
          << "[" << category "] "

namespace {

/* We can create Arrow buffers which refer memory owned by ResultSet.
   For safe memory access we should keep a ResultSetPtr to keep
   data live while buffer lives. Use this custom buffer for that. */
class ResultSetBuffer : public arrow::Buffer {
 public:
  ResultSetBuffer(const uint8_t* buf, size_t size, ResultSetPtr rs)
      : arrow::Buffer(buf, size), _rs(rs) {}

 private:
  ResultSetPtr _rs;
};

template <typename TYPE, typename VALUE_ARRAY_TYPE>
void create_or_append_value(const ScalarTargetValue& val_cty,
                            std::shared_ptr<ValueArray>& values,
                            const size_t max_size) {
  auto pval_cty = boost::get<VALUE_ARRAY_TYPE>(&val_cty);
  CHECK(pval_cty);
  auto val_ty = static_cast<TYPE>(*pval_cty);
  if (!values) {
    values = std::make_shared<ValueArray>(std::vector<TYPE>());
    boost::get<std::vector<TYPE>>(*values).reserve(max_size);
  }
  CHECK(values);
  auto values_ty = boost::get<std::vector<TYPE>>(values.get());
  CHECK(values_ty);
  values_ty->push_back(val_ty);
}

template <>
void create_or_append_value<std::string, NullableString>(
    const ScalarTargetValue& val_cty,
    std::shared_ptr<ValueArray>& values,
    const size_t max_size) {
  auto pval_cty = boost::get<NullableString>(&val_cty);
  CHECK(pval_cty);
  std::string val_ty;
  if (pval_cty->type() != typeid(void*)) {
    val_ty = boost::get<std::string>(*pval_cty);
  }
  if (!values) {
    values = std::make_shared<ValueArray>(std::vector<std::string>());
    boost::get<std::vector<std::string>>(*values).reserve(max_size);
  }
  CHECK(values);
  auto values_ty = boost::get<std::vector<std::string>>(values.get());
  CHECK(values_ty);
  values_ty->push_back(val_ty);
}

template <typename TYPE>
void create_or_append_validity(const ScalarTargetValue& value,
                               const hdk::ir::Type* col_type,
                               std::shared_ptr<std::vector<bool>>& null_bitmap,
                               const size_t max_size) {
  if (!col_type->nullable()) {
    CHECK(!null_bitmap);
    return;
  }
  auto pvalue = boost::get<TYPE>(&value);
  CHECK(pvalue);
  bool is_valid = false;
  if (col_type->isBoolean()) {
    is_valid = inline_int_null_value(col_type) != static_cast<int8_t>(*pvalue);
  } else if (col_type->isExtDictionary()) {
    is_valid = inline_int_null_value(col_type) != static_cast<int32_t>(*pvalue);
  } else if (col_type->isInteger() || col_type->isDateTime()) {
    is_valid = inline_int_null_value(col_type) != static_cast<int64_t>(*pvalue);
  } else if (col_type->isFloatingPoint()) {
    is_valid = inline_fp_null_value(col_type) != static_cast<double>(*pvalue);
  } else if (col_type->isDecimal()) {
    is_valid = inline_int_null_value(col_type) != static_cast<int64_t>(*pvalue);
  } else {
    UNREACHABLE();
  }

  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(is_valid);
}

template <>
void create_or_append_validity<NullableString>(
    const ScalarTargetValue& value,
    const hdk::ir::Type* col_type,
    std::shared_ptr<std::vector<bool>>& null_bitmap,
    const size_t max_size) {
  if (!col_type->nullable()) {
    CHECK(!null_bitmap);
    return;
  }
  auto pvalue = boost::get<NullableString>(&value);
  CHECK(pvalue);
  bool is_valid = false;
  if (col_type->isText()) {
    is_valid = pvalue->type() != typeid(void*);
  } else {
    UNREACHABLE();
  }

  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(is_valid);
}

template <typename TYPE, typename enable = void>
class null_type {};

template <typename TYPE>
struct null_type<TYPE, std::enable_if_t<std::is_integral<TYPE>::value>> {
  using type = typename std::make_signed<TYPE>::type;
  static constexpr type value = inline_int_null_value<type>();
};

template <typename TYPE>
struct null_type<TYPE, std::enable_if_t<std::is_floating_point<TYPE>::value>> {
  using type = TYPE;
  static constexpr type value = inline_fp_null_value<type>();
};

template <typename TYPE>
using null_type_t = typename null_type<TYPE>::type;

template <typename C_TYPE,
          typename ARROW_TYPE = typename arrow::CTypeTraits<C_TYPE>::ArrowType>
void convert_column(ResultSetPtr result,
                    size_t col,
                    size_t entry_count,
                    std::shared_ptr<arrow::Array>& out) {
  CHECK(sizeof(C_TYPE) == result->colType(col)->size());

  std::shared_ptr<arrow::Buffer> values;
  std::shared_ptr<arrow::Buffer> is_valid;
  const int64_t buf_size = entry_count * sizeof(C_TYPE);
  if (result->isZeroCopyColumnarConversionPossible(col)) {
    values.reset(new ResultSetBuffer(
        reinterpret_cast<const uint8_t*>(result->getColumnarBuffer(col)),
        buf_size,
        result));
  } else {
    auto res = arrow::AllocateBuffer(buf_size);
    CHECK(res.ok());
    values = std::move(res).ValueOrDie();
    result->copyColumnIntoBuffer(
        col, reinterpret_cast<int8_t*>(values->mutable_data()), buf_size);
  }

  int64_t null_count = 0;
  auto res = arrow::AllocateBuffer((entry_count + 7) / 8);
  CHECK(res.ok());
  is_valid = std::move(res).ValueOrDie();

  auto is_valid_data = is_valid->mutable_data();

  const null_type_t<C_TYPE>* vals =
      reinterpret_cast<const null_type_t<C_TYPE>*>(values->data());
  null_type_t<C_TYPE> null_val = null_type<C_TYPE>::value;

  size_t unroll_count = entry_count & 0xFFFFFFFFFFFFFFF8ULL;
  for (size_t i = 0; i < unroll_count; i += 8) {
    uint8_t valid_byte = 0;
    uint8_t valid;
    valid = vals[i + 0] != null_val;
    valid_byte |= valid << 0;
    null_count += !valid;
    valid = vals[i + 1] != null_val;
    valid_byte |= valid << 1;
    null_count += !valid;
    valid = vals[i + 2] != null_val;
    valid_byte |= valid << 2;
    null_count += !valid;
    valid = vals[i + 3] != null_val;
    valid_byte |= valid << 3;
    null_count += !valid;
    valid = vals[i + 4] != null_val;
    valid_byte |= valid << 4;
    null_count += !valid;
    valid = vals[i + 5] != null_val;
    valid_byte |= valid << 5;
    null_count += !valid;
    valid = vals[i + 6] != null_val;
    valid_byte |= valid << 6;
    null_count += !valid;
    valid = vals[i + 7] != null_val;
    valid_byte |= valid << 7;
    null_count += !valid;
    is_valid_data[i >> 3] = valid_byte;
  }
  if (unroll_count != entry_count) {
    uint8_t valid_byte = 0;
    for (size_t i = unroll_count; i < entry_count; ++i) {
      bool valid = vals[i] != null_val;
      valid_byte |= valid << (i & 7);
      null_count += !valid;
    }
    is_valid_data[unroll_count >> 3] = valid_byte;
  }

  if (!null_count) {
    is_valid.reset();
  }

  // TODO: support date/time + scaling
  // TODO: support booleans
  if (null_count) {
    out.reset(
        new arrow::NumericArray<ARROW_TYPE>(entry_count, values, is_valid, null_count));
  } else {
    out.reset(new arrow::NumericArray<ARROW_TYPE>(entry_count, values));
  }
}

template <typename TYPE>
size_t gen_bitmap(uint8_t* bitmap, const TYPE* data, size_t size) {
  static_assert(
      sizeof(TYPE) == 1 || sizeof(TYPE) == 2 || sizeof(TYPE) == 4 || sizeof(TYPE) == 8,
      "gen_bitmap() -- Size of TYPE must be 1, 2, 4, or 8.");
  TYPE nullval = inline_null_value<TYPE>();

  size_t rv = 0;
  if constexpr (sizeof(TYPE) == 1) {
    rv = gen_null_bitmap_8(bitmap,
                           reinterpret_cast<const uint8_t*>(data),
                           size,
                           reinterpret_cast<const uint8_t&>(nullval));
  } else if constexpr (sizeof(TYPE) == 2) {
    rv = gen_null_bitmap_16(bitmap,
                            reinterpret_cast<const uint16_t*>(data),
                            size,
                            reinterpret_cast<const uint16_t&>(nullval));
  } else if constexpr (sizeof(TYPE) == 4) {
    uint32_t casted_nullval;
    memcpy(&casted_nullval, &nullval, 4);
    rv = gen_null_bitmap_32(
        bitmap, reinterpret_cast<const uint32_t*>(data), size, casted_nullval);
  } else if constexpr (sizeof(TYPE) == 8) {
    uint64_t casted_nullval;
    memcpy(&casted_nullval, &nullval, 8);
    rv = gen_null_bitmap_64(
        bitmap, reinterpret_cast<const uint64_t*>(data), size, casted_nullval);
  }
  return rv;
}

template <typename TYPE>
int64_t create_bitmap_parallel_for_avx512(uint8_t* bitmap_data,
                                          const TYPE* vals,
                                          const size_t vals_size) {
  static_assert(sizeof(TYPE) <= 64 && (64 % sizeof(TYPE) == 0),
                "Size of type must not exceed 64 and should divide 64.");

  std::atomic<int64_t> null_count = 0;

  constexpr size_t min_block_size = 64ULL / sizeof(TYPE);
  const size_t cpu_processing_count = vals_size % min_block_size;
  const size_t avx512_processing_count = vals_size - cpu_processing_count;

  auto br_par_processor = [&](const tbb::blocked_range<size_t>& r) {
    size_t idx = min_block_size * r.begin();
    size_t processing_count =
        std::min(min_block_size * r.end() - idx, avx512_processing_count - idx);
    uint8_t* bitmap_data_ptr = bitmap_data + idx / 8;
    const TYPE* values_data_ptr = vals + idx;
    null_count += gen_bitmap<TYPE>(
        bitmap_data_ptr, const_cast<TYPE*>(values_data_ptr), processing_count);
  };

  threading::parallel_for(
      tbb::blocked_range<size_t>(
          0, (avx512_processing_count + min_block_size - 1) / min_block_size),
      br_par_processor);

  if (cpu_processing_count > 0) {
    TYPE null_val = inline_null_value<TYPE>();

    size_t remaining_bits = 0;
    int64_t cpus_null_count = 0;
    for (size_t i = 0; i < cpu_processing_count; ++i) {
      size_t valid = vals[avx512_processing_count + i] != null_val;
      remaining_bits |= valid << i;
      cpus_null_count += !valid;
    }

    size_t left_bytes_encoded_count = (cpu_processing_count + 7) / 8;
    for (size_t i = 0; i < left_bytes_encoded_count; i++) {
      uint8_t encoded_byte = 0xFF & (remaining_bits >> (8 * i));
      bitmap_data[avx512_processing_count / 8 + i] = encoded_byte;
    }
    null_count += cpus_null_count;
  }

  return null_count.load();
}

// convert_column() specialization for arrow::ChunkedArray output
template <typename C_TYPE,
          typename ARROW_TYPE = typename arrow::CTypeTraits<C_TYPE>::ArrowType>
void convert_column(ResultSetPtr result,
                    size_t col,
                    size_t entry_count,
                    std::shared_ptr<arrow::ChunkedArray>& out) {
  CHECK(sizeof(C_TYPE) == result->colType(col)->size());

  std::vector<std::shared_ptr<arrow::Buffer>> values;

  CHECK(result->isChunkedZeroCopyColumnarConversionPossible(col));

  auto chunks = result->getChunkedColumnarBuffer(col);
  size_t total_row_count = 0;
  for (auto& [chunk_ptr, chunk_rows_count] : chunks) {
    const int64_t buf_size = chunk_rows_count * sizeof(C_TYPE);
    total_row_count += chunk_rows_count;
    values.emplace_back(new ResultSetBuffer(
        reinterpret_cast<const uint8_t*>(chunk_ptr), buf_size, result));
  }

  CHECK_LE(total_row_count, entry_count);

  std::vector<std::shared_ptr<arrow::Array>> fragments(values.size(), nullptr);

  threading::parallel_for(static_cast<size_t>(0), values.size(), [&](size_t idx) {
    size_t chunk_rows_count = chunks[idx].second;

    auto res = arrow::AllocateBuffer((chunk_rows_count + 7) / 8);
    CHECK(res.ok());

    std::shared_ptr<arrow::Buffer> is_valid = std::move(res).ValueOrDie();

    uint8_t* bitmap = is_valid->mutable_data();

    const null_type_t<C_TYPE>* vals =
        reinterpret_cast<const null_type_t<C_TYPE>*>(values[idx]->data());

    int64_t null_count = create_bitmap_parallel_for_avx512<null_type_t<C_TYPE>>(
        bitmap, vals, chunk_rows_count);

    if (!null_count) {
      is_valid.reset();
    }

    // TODO: support date/time + scaling
    // TODO: support booleans
    using NumArray = arrow::NumericArray<ARROW_TYPE>;
    fragments[idx] = null_count
                         ? std::make_shared<NumArray>(
                               chunk_rows_count, values[idx], is_valid, null_count)
                         : std::make_shared<NumArray>(chunk_rows_count, values[idx]);
  });  // threading::parallel_for

  out = std::make_shared<arrow::ChunkedArray>(std::move(fragments));
}

template <typename ArrowArrayType>
void convert_column(const hdk::ir::Type* physical_type,
                    ResultSetPtr results,
                    size_t col_idx,
                    size_t entry_count,
                    std::shared_ptr<ArrowArrayType>& out) {
  switch (physical_type->id()) {
    case hdk::ir::Type::kInteger:
      switch (physical_type->size()) {
        case 1:
          convert_column<int8_t>(results, col_idx, entry_count, out);
          break;
        case 2:
          convert_column<int16_t>(results, col_idx, entry_count, out);
          break;
        case 4:
          convert_column<int32_t>(results, col_idx, entry_count, out);
          break;
        case 8:
          convert_column<int64_t>(results, col_idx, entry_count, out);
          break;
        default:
          throw std::runtime_error(physical_type->toString() +
                                   " is not supported in Arrow column converter.");
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (physical_type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          convert_column<float>(results, col_idx, entry_count, out);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          convert_column<double>(results, col_idx, entry_count, out);
          break;
        default:
          throw std::runtime_error(physical_type->toString() +
                                   " is not supported in Arrow column converter.");
      }
      break;
    default:
      throw std::runtime_error(physical_type->toString() +
                               " is not supported in Arrow column converter.");
  }
}

#ifndef _MSC_VER
std::pair<key_t, void*> get_shm(size_t shmsz) {
  if (!shmsz) {
    return std::make_pair(IPC_PRIVATE, nullptr);
  }
  // Generate a new key for a shared memory segment. Keys to shared memory segments
  // are OS global, so we need to try a new key if we encounter a collision. It seems
  // incremental keygen would be deterministically worst-case. If we use a hash
  // (like djb2) + nonce, we could still get collisions if multiple clients specify
  // the same nonce, so using rand() in lieu of a better approach
  // TODO(ptaylor): Is this common? Are these assumptions true?
  auto key = static_cast<key_t>(rand());
  int shmid = -1;
  // IPC_CREAT - indicates we want to create a new segment for this key if it doesn't
  // exist IPC_EXCL - ensures failure if a segment already exists for this key
  while ((shmid = shmget(key, shmsz, IPC_CREAT | IPC_EXCL | 0666)) < 0) {
    // If shmget fails and errno is one of these four values, try a new key.
    // TODO(ptaylor): is checking for the last three values really necessary? Checking
    // them by default to be safe. EEXIST - a shared memory segment is already associated
    // with this key EACCES - a shared memory segment is already associated with this key,
    // but we don't have permission to access it EINVAL - a shared memory segment is
    // already associated with this key, but the size is less than shmsz ENOENT -
    // IPC_CREAT was not set in shmflg and no shared memory segment associated with key
    // was found
    if (!(errno & (EEXIST | EACCES | EINVAL | ENOENT))) {
      throw std::runtime_error("failed to create a shared memory");
    }
    key = static_cast<key_t>(rand());
  }
  // get a pointer to the shared memory segment
  auto ipc_ptr = shmat(shmid, NULL, 0);
  if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
    throw std::runtime_error("failed to attach a shared memory");
  }

  return std::make_pair(key, ipc_ptr);
}
#endif

std::pair<key_t, std::shared_ptr<arrow::Buffer>> get_shm_buffer(size_t size) {
#ifdef _MSC_VER
  throw std::runtime_error("Arrow IPC not yet supported on Windows.");
  return std::make_pair(0, nullptr);
#else
  auto [key, ipc_ptr] = get_shm(size);
  std::shared_ptr<arrow::Buffer> buffer(
      new arrow::MutableBuffer(static_cast<uint8_t*>(ipc_ptr), size));
  return std::make_pair<key_t, std::shared_ptr<arrow::Buffer>>(std::move(key),
                                                               std::move(buffer));
#endif
}

}  // anonymous namespace

namespace arrow {

key_t get_and_copy_to_shm(const std::shared_ptr<Buffer>& data) {
#ifdef _MSC_VER
  throw std::runtime_error("Arrow IPC not yet supported on Windows.");
#else
  auto [key, ipc_ptr] = get_shm(data->size());
  // copy the arrow records buffer to shared memory
  // TODO(ptaylor): I'm sure it's possible to tell Arrow's RecordBatchStreamWriter to
  // write directly to the shared memory segment as a sink
  memcpy(ipc_ptr, data->data(), data->size());
  // detach from the shared memory segment
  shmdt(ipc_ptr);
  return key;
#endif
}

}  // namespace arrow

//! Serialize an Arrow result to IPC memory. Users are responsible for freeing all CPU IPC
//! buffers using deallocateArrowResultBuffer. GPU buffers will become owned by the caller
//! upon deserialization, and will be automatically freed when they go out of scope.
ArrowResult ArrowResultSetConverter::getArrowResult() const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> record_batch = convertToArrow();

  struct BuildResultParams {
    int64_t schemaSize() const {
      return serialized_schema ? serialized_schema->size() : 0;
    };
    int64_t dictSize() const { return serialized_dict ? serialized_dict->size() : 0; };
    int64_t totalSize() const { return schemaSize() + records_size + dictSize(); }
    bool hasRecordBatch() const { return records_size > 0; }
    bool hasDict() const { return dictSize() > 0; }

    int64_t records_size{0};
    std::shared_ptr<arrow::Buffer> serialized_schema{nullptr};
    std::shared_ptr<arrow::Buffer> serialized_dict{nullptr};
  } result_params;

  if (device_type_ == ExecutorDeviceType::CPU ||
      transport_method_ == ArrowTransport::WIRE) {
    const auto getWireResult = [&]() -> ArrowResult {
      auto timer = DEBUG_TIMER("serialize batch to wire");
      const auto total_size = result_params.totalSize();
      std::vector<char> record_handle_data(total_size);
      auto serialized_records =
          arrow::MutableBuffer::Wrap(record_handle_data.data(), total_size);

      ARROW_ASSIGN_OR_THROW(auto writer, arrow::Buffer::GetWriter(serialized_records));

      ARROW_THROW_NOT_OK(writer->Write(
          reinterpret_cast<const uint8_t*>(result_params.serialized_schema->data()),
          result_params.schemaSize()));

      if (result_params.hasDict()) {
        ARROW_THROW_NOT_OK(writer->Write(
            reinterpret_cast<const uint8_t*>(result_params.serialized_dict->data()),
            result_params.dictSize()));
      }

      arrow::io::FixedSizeBufferWriter stream(SliceMutableBuffer(
          serialized_records, result_params.schemaSize() + result_params.dictSize()));

      if (result_params.hasRecordBatch()) {
        ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(
            *record_batch, arrow::ipc::IpcWriteOptions::Defaults(), &stream));
      }

      return {std::vector<char>(0),
              0,
              std::vector<char>(0),
              serialized_records->size(),
              std::string{""},
              std::move(record_handle_data)};
    };

    const auto getShmResult = [&]() -> ArrowResult {
      auto timer = DEBUG_TIMER("serialize batch to shared memory");
      std::shared_ptr<arrow::Buffer> serialized_records;
      std::vector<char> schema_handle_buffer;
      std::vector<char> record_handle_buffer(sizeof(key_t), 0);
      key_t records_shm_key = IPC_PRIVATE;
      const int64_t total_size = result_params.totalSize();

      std::tie(records_shm_key, serialized_records) = get_shm_buffer(total_size);

      memcpy(serialized_records->mutable_data(),
             result_params.serialized_schema->data(),
             (size_t)result_params.schemaSize());

      if (result_params.hasDict()) {
        memcpy(serialized_records->mutable_data() + result_params.schemaSize(),
               result_params.serialized_dict->data(),
               (size_t)result_params.dictSize());
      }

      arrow::io::FixedSizeBufferWriter stream(SliceMutableBuffer(
          serialized_records, result_params.schemaSize() + result_params.dictSize()));

      if (result_params.hasRecordBatch()) {
        ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(
            *record_batch, arrow::ipc::IpcWriteOptions::Defaults(), &stream));
      }

      memcpy(&record_handle_buffer[0],
             reinterpret_cast<const unsigned char*>(&records_shm_key),
             sizeof(key_t));

      return {schema_handle_buffer,
              0,
              record_handle_buffer,
              serialized_records->size(),
              std::string{""}};
    };

    arrow::ipc::DictionaryFieldMapper mapper(*record_batch->schema());
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    auto dict_stream = arrow::io::BufferOutputStream::Create(1024).ValueOrDie();

    // If our record batch is going to be empty, we omit it entirely,
    // only serializing the schema.
    if (!record_batch->num_rows()) {
      ARROW_ASSIGN_OR_THROW(result_params.serialized_schema,
                            arrow::ipc::SerializeSchema(*record_batch->schema(),
                                                        arrow::default_memory_pool()));

      switch (transport_method_) {
        case ArrowTransport::WIRE:
          return getWireResult();
        case ArrowTransport::SHARED_MEMORY:
          return getShmResult();
        default:
          UNREACHABLE();
      }
    }

    ARROW_ASSIGN_OR_THROW(auto dictionaries, CollectDictionaries(*record_batch, mapper));

    ARROW_LOG("CPU") << "found " << dictionaries.size() << " dictionaries";

    for (auto& pair : dictionaries) {
      arrow::ipc::IpcPayload payload;
      int64_t dictionary_id = pair.first;
      const auto& dictionary = pair.second;

      ARROW_THROW_NOT_OK(
          GetDictionaryPayload(dictionary_id, dictionary, options, &payload));
      int32_t metadata_length = 0;
      ARROW_THROW_NOT_OK(
          WriteIpcPayload(payload, options, dict_stream.get(), &metadata_length));
    }
    result_params.serialized_dict = dict_stream->Finish().ValueOrDie();

    ARROW_ASSIGN_OR_THROW(result_params.serialized_schema,
                          arrow::ipc::SerializeSchema(*record_batch->schema(),
                                                      arrow::default_memory_pool()));

    ARROW_THROW_NOT_OK(
        arrow::ipc::GetRecordBatchSize(*record_batch, &result_params.records_size));

    switch (transport_method_) {
      case ArrowTransport::WIRE:
        return getWireResult();
      case ArrowTransport::SHARED_MEMORY:
        return getShmResult();
      default:
        UNREACHABLE();
    }
  }
#ifdef HAVE_CUDA
  CHECK(device_type_ == ExecutorDeviceType::GPU);

  // Copy the schema to the schema handle
  auto out_stream_result = arrow::io::BufferOutputStream::Create(1024);
  ARROW_THROW_NOT_OK(out_stream_result.status());
  auto out_stream = std::move(out_stream_result).ValueOrDie();

  arrow::ipc::DictionaryFieldMapper mapper(*record_batch->schema());
  arrow::ipc::DictionaryMemo current_memo;
  arrow::ipc::DictionaryMemo serialized_memo;

  arrow::ipc::IpcPayload schema_payload;
  ARROW_THROW_NOT_OK(arrow::ipc::GetSchemaPayload(*record_batch->schema(),
                                                  arrow::ipc::IpcWriteOptions::Defaults(),
                                                  mapper,
                                                  &schema_payload));
  int32_t schema_payload_length = 0;
  ARROW_THROW_NOT_OK(arrow::ipc::WriteIpcPayload(schema_payload,
                                                 arrow::ipc::IpcWriteOptions::Defaults(),
                                                 out_stream.get(),
                                                 &schema_payload_length));
  ARROW_ASSIGN_OR_THROW(auto dictionaries, CollectDictionaries(*record_batch, mapper));
  ARROW_LOG("GPU") << "Dictionary "
                   << "found dicts: " << dictionaries.size();

  ARROW_THROW_NOT_OK(
      arrow::ipc::internal::CollectDictionaries(*record_batch, &current_memo));

  // now try a dictionary
  std::shared_ptr<arrow::Schema> dummy_schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dict_batches;

  for (const auto& pair : dictionaries) {
    arrow::ipc::IpcPayload payload;
    const auto& dict_id = pair.first;
    CHECK_GE(dict_id, 0);
    ARROW_LOG("GPU") << "Dictionary "
                     << "dict_id: " << dict_id;
    const auto& dict = pair.second;
    CHECK(dict);

    if (!dummy_schema) {
      auto dummy_field = std::make_shared<arrow::Field>("", dict->type());
      dummy_schema = std::make_shared<arrow::Schema>(
          std::vector<std::shared_ptr<arrow::Field>>{dummy_field});
    }
    dict_batches.emplace_back(
        arrow::RecordBatch::Make(dummy_schema, dict->length(), {dict}));
  }

  if (!dict_batches.empty()) {
    ARROW_THROW_NOT_OK(arrow::ipc::WriteRecordBatchStream(
        dict_batches, arrow::ipc::IpcWriteOptions::Defaults(), out_stream.get()));
  }

  auto complete_ipc_stream = out_stream->Finish();
  ARROW_THROW_NOT_OK(complete_ipc_stream.status());
  auto serialized_records = std::move(complete_ipc_stream).ValueOrDie();

  const auto record_key = arrow::get_and_copy_to_shm(serialized_records);
  std::vector<char> schema_record_key_buffer(sizeof(key_t), 0);
  memcpy(&schema_record_key_buffer[0],
         reinterpret_cast<const unsigned char*>(&record_key),
         sizeof(key_t));

  arrow::cuda::CudaDeviceManager* manager;
  ARROW_ASSIGN_OR_THROW(manager, arrow::cuda::CudaDeviceManager::Instance());
  std::shared_ptr<arrow::cuda::CudaContext> context;
  ARROW_ASSIGN_OR_THROW(context, manager->GetContext(device_id_));

  std::shared_ptr<arrow::cuda::CudaBuffer> device_serialized;
  ARROW_ASSIGN_OR_THROW(device_serialized,
                        SerializeRecordBatch(*record_batch, context.get()));

  std::shared_ptr<arrow::cuda::CudaIpcMemHandle> cuda_handle;
  ARROW_ASSIGN_OR_THROW(cuda_handle, device_serialized->ExportForIpc());

  std::shared_ptr<arrow::Buffer> serialized_cuda_handle;
  ARROW_ASSIGN_OR_THROW(serialized_cuda_handle,
                        cuda_handle->Serialize(arrow::default_memory_pool()));

  std::vector<char> record_handle_buffer(serialized_cuda_handle->size(), 0);
  memcpy(&record_handle_buffer[0],
         serialized_cuda_handle->data(),
         serialized_cuda_handle->size());

  return {schema_record_key_buffer,
          serialized_records->size(),
          record_handle_buffer,
          serialized_cuda_handle->size(),
          serialized_cuda_handle->ToString()};
#else
  UNREACHABLE();
  return {std::vector<char>{}, 0, std::vector<char>{}, 0, ""};
#endif
}

ArrowResultSetConverter::SerializedArrowOutput
ArrowResultSetConverter::getSerializedArrowOutput(
    arrow::ipc::DictionaryFieldMapper* mapper) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> arrow_copy = convertToArrow();
  std::shared_ptr<arrow::Buffer> serialized_records, serialized_schema;

  ARROW_ASSIGN_OR_THROW(
      serialized_schema,
      arrow::ipc::SerializeSchema(*arrow_copy->schema(), arrow::default_memory_pool()));

  if (arrow_copy->num_rows()) {
    auto timer = DEBUG_TIMER("serialize records");
    ARROW_THROW_NOT_OK(arrow_copy->Validate());
    ARROW_ASSIGN_OR_THROW(serialized_records,
                          arrow::ipc::SerializeRecordBatch(
                              *arrow_copy, arrow::ipc::IpcWriteOptions::Defaults()));
  } else {
    ARROW_ASSIGN_OR_THROW(serialized_records, arrow::AllocateBuffer(0));
  }
  return {serialized_schema, serialized_records};
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::convertToArrow() const {
  auto timer = DEBUG_TIMER(__func__);
  return getArrowBatch(makeSchema());
}

std::shared_ptr<arrow::Table> ArrowResultSetConverter::convertToArrowTable() const {
  auto timer = DEBUG_TIMER(__func__);
  return getArrowTable(makeSchema());
}

void append_scalar_value_and_validity(const ScalarTargetValue& value,
                                      const hdk::ir::Type* type,
                                      ExecutorDeviceType device_type,
                                      std::shared_ptr<ValueArray>& values,
                                      std::shared_ptr<std::vector<bool>>& null_bitmap,
                                      size_t max_size) {
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      create_or_append_value<bool, int64_t>(value, values, max_size);
      create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
      break;
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kExtDictionary:
      switch (type->size()) {
        case 1:
          create_or_append_value<int8_t, int64_t>(value, values, max_size);
          create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
          break;
        case 2:
          create_or_append_value<int16_t, int64_t>(value, values, max_size);
          create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
          break;
        case 4:
          create_or_append_value<int32_t, int64_t>(value, values, max_size);
          create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
          break;
        case 8:
          create_or_append_value<int64_t, int64_t>(value, values, max_size);
          create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
          break;
        default:
          throw std::runtime_error(type->toString() +
                                   " is not supported in Arrow result sets.");
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          create_or_append_value<float, float>(value, values, max_size);
          create_or_append_validity<float>(value, type, null_bitmap, max_size);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          create_or_append_value<double, double>(value, values, max_size);
          create_or_append_validity<double>(value, type, null_bitmap, max_size);
          break;
        default:
          throw std::runtime_error(type->toString() +
                                   " is not supported in Arrow result sets.");
      }
      break;
    case hdk::ir::Type::kTime:
      create_or_append_value<int32_t, int64_t>(value, values, max_size);
      create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
      break;
    case hdk::ir::Type::kDate:
      device_type == ExecutorDeviceType::GPU
          ? create_or_append_value<int64_t, int64_t>(value, values, max_size)
          : create_or_append_value<int32_t, int64_t>(value, values, max_size);
      create_or_append_validity<int64_t>(value, type, null_bitmap, max_size);
      break;
    case hdk::ir::Type::kText:
    case hdk::ir::Type::kVarChar:
      create_or_append_value<std::string, NullableString>(value, values, max_size);
      create_or_append_validity<NullableString>(value, type, null_bitmap, max_size);
      break;
    default:
      // TODO(miyu): support more scalar types.
      throw std::runtime_error(type->toString() +
                               " is not supported in Arrow result sets.");
  }
}

size_t convert_rowwise(
    ResultSetPtr results,
    const std::vector<ArrowResultSetConverter::ColumnBuilder>& builders,
    ExecutorDeviceType device_type,
    std::vector<std::shared_ptr<ValueArray>>& value_seg,
    std::vector<std::shared_ptr<ValueArray>>& offset_seg,
    std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
    std::vector<std::shared_ptr<std::vector<uint8_t>>>& offset_null_bitmap_seg,
    const std::vector<bool>& non_lazy_cols,
    const size_t start_entry,
    const size_t end_entry,
    bool is_truncated = false) {
  const auto col_count = results->colCount();
  CHECK_EQ(value_seg.size(), col_count);
  CHECK_EQ(null_bitmap_seg.size(), col_count);
  const auto local_entry_count = end_entry - start_entry;
  size_t seg_row_count = 0;
  size_t limit = results->getLimit();
  size_t offset = results->getOffset();
  for (size_t i = start_entry; i < end_entry; ++i) {
    if (is_truncated && seg_row_count >= offset + limit) {
      break;
    }

    auto row = results->getRowAtNoTranslations(i, non_lazy_cols);
    if (row.empty()) {
      continue;
    }
    ++seg_row_count;

    if (is_truncated && seg_row_count <= offset) {
      continue;
    }

    for (size_t j = 0; j < col_count; ++j) {
      if (!non_lazy_cols.empty() && non_lazy_cols[j]) {
        continue;
      }

      const auto& column = builders[j];
      auto array_value = boost::get<ArrayTargetValue>(&row[j]);
      if (array_value) {
        auto elem_type = column.col_type->as<hdk::ir::VarLenArrayType>()->elemType();
        int32_t arr_size = 0;
        if (*array_value) {
          arr_size = static_cast<int32_t>((*array_value)->size());
          for (auto& elem_value : **array_value) {
            append_scalar_value_and_validity(elem_value,
                                             elem_type,
                                             device_type,
                                             value_seg[j],
                                             null_bitmap_seg[j],
                                             local_entry_count);
          }
        }

        auto& offsets = offset_seg[j];
        if (!offsets) {
          offsets = std::make_shared<ValueArray>(std::vector<int32_t>());
          boost::get<std::vector<int32_t>>(*offsets).reserve(local_entry_count);
          boost::get<std::vector<int32_t>>(*offsets).push_back(0);
        }
        auto& offset_values = boost::get<std::vector<int32_t>>(*offsets);
        offset_values.push_back(offset_values.back() + arr_size);

        if (column.col_type->nullable()) {
          auto& validity = offset_null_bitmap_seg[j];
          if (!validity) {
            validity = std::make_shared<std::vector<uint8_t>>();
            validity->reserve(local_entry_count);
          }
          validity->push_back(!!*array_value);
        }
      } else {
        auto scalar_value = boost::get<ScalarTargetValue>(&row[j]);
        CHECK(scalar_value);
        append_scalar_value_and_validity(*scalar_value,
                                         column.col_type,
                                         device_type,
                                         value_seg[j],
                                         null_bitmap_seg[j],
                                         local_entry_count);
      }
    }
  }
  return seg_row_count;
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::getArrowBatch(
    const std::shared_ptr<arrow::Schema>& schema) const {
  std::vector<std::shared_ptr<arrow::Array>> result_columns;

  // First, check if the result set is empty.
  // If so, we return an arrow result set that only
  // contains the schema (no record batch will be serialized).
  if (results_->isEmpty()) {
    return ARROW_RECORDBATCH_MAKE(schema, 0, result_columns);
  }

  const size_t entry_count = top_n_ < 0
                                 ? results_->entryCount()
                                 : std::min(size_t(top_n_), results_->entryCount());

  const auto col_count = results_->colCount();
  size_t row_count = 0;

  result_columns.resize(col_count);
  std::vector<ColumnBuilder> builders(col_count);

  // Create array builders
  for (size_t i = 0; i < col_count; ++i) {
    initializeColumnBuilder(builders[i], results_->colType(i), i, schema->field(i));
  }

  // TODO(miyu): speed up for columnar buffers
  auto fetch =
      [&](std::vector<std::shared_ptr<ValueArray>>& value_seg,
          std::vector<std::shared_ptr<ValueArray>>& offset_seg,
          std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
          std::vector<std::shared_ptr<std::vector<uint8_t>>>& null_bitmap_offset_seg,
          const std::vector<bool>& non_lazy_cols,
          const size_t start_entry,
          const size_t end_entry) -> size_t {
    return convert_rowwise(results_,
                           builders,
                           device_type_,
                           value_seg,
                           offset_seg,
                           null_bitmap_seg,
                           null_bitmap_offset_seg,
                           non_lazy_cols,
                           start_entry,
                           end_entry);
  };

  auto convert_columns = [&](std::vector<std::shared_ptr<arrow::Array>>& result,
                             const std::vector<bool>& non_lazy_cols,
                             const size_t start_col,
                             const size_t end_col) {
    for (size_t col = start_col; col < end_col; ++col) {
      if (!non_lazy_cols.empty() && !non_lazy_cols[col]) {
        continue;
      }

      convert_column(
          builders[col].physical_type, results_, col, entry_count, result[col]);
    }
  };

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<ValueArray>> column_offsets(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<uint8_t>>> offset_null_bitmaps(col_count,
                                                                         nullptr);
  const bool multithreaded = entry_count > 10000 && !results_->isTruncated();
  bool use_columnar_converter = results_->isDirectColumnarConversionPossible() &&
                                results_->getQueryMemDesc().getQueryDescriptionType() ==
                                    QueryDescriptionType::Projection &&
                                entry_count == results_->entryCount();
  std::vector<bool> non_lazy_cols;
  if (use_columnar_converter) {
    auto timer = DEBUG_TIMER("columnar converter");
    std::vector<size_t> non_lazy_col_pos;
    size_t non_lazy_col_count = 0;
    const auto& lazy_fetch_info = results_->getLazyFetchInfo();

    non_lazy_cols.reserve(col_count);
    non_lazy_col_pos.reserve(col_count);
    for (size_t i = 0; i < col_count; ++i) {
      bool is_lazy =
          lazy_fetch_info.empty() ? false : lazy_fetch_info[i].is_lazily_fetched;
      // Currently column converter cannot handle some data types.
      // Treat them as lazy.
      switch (builders[i].physical_type->id()) {
        case hdk::ir::Type::kBoolean:
        case hdk::ir::Type::kTime:
        case hdk::ir::Type::kDate:
        case hdk::ir::Type::kTimestamp:
          is_lazy = true;
          break;
        default:
          break;
      }
      if (builders[i].field->type()->id() == arrow::Type::DICTIONARY) {
        is_lazy = true;
      }
      non_lazy_cols.emplace_back(!is_lazy);
      if (!is_lazy) {
        ++non_lazy_col_count;
        non_lazy_col_pos.emplace_back(i);
      }
    }

    if (non_lazy_col_count == col_count) {
      non_lazy_cols.clear();
      non_lazy_col_pos.clear();
    } else {
      non_lazy_col_pos.emplace_back(col_count);
    }

    std::vector<std::future<void>> child_threads;
    size_t num_threads =
        std::min(multithreaded ? (size_t)cpu_threads() : (size_t)1, non_lazy_col_count);

    size_t start_col = 0;
    size_t end_col = 0;
    for (size_t i = 0; i < num_threads; ++i) {
      start_col = end_col;
      end_col = (i + 1) * non_lazy_col_count / num_threads;
      size_t phys_start_col =
          non_lazy_col_pos.empty() ? start_col : non_lazy_col_pos[start_col];
      size_t phys_end_col =
          non_lazy_col_pos.empty() ? end_col : non_lazy_col_pos[end_col];
      child_threads.push_back(std::async(std::launch::async,
                                         convert_columns,
                                         std::ref(result_columns),
                                         non_lazy_cols,
                                         phys_start_col,
                                         phys_end_col));
    }
    for (auto& child : child_threads) {
      child.get();
    }
    row_count = entry_count;
  }
  if (!use_columnar_converter || !non_lazy_cols.empty()) {
    auto timer = DEBUG_TIMER("row converter");
    row_count = 0;
    if (multithreaded) {
      const size_t cpu_count = cpu_threads();
      std::vector<std::future<size_t>> child_threads;
      std::vector<std::vector<std::shared_ptr<ValueArray>>> column_value_segs(
          cpu_count, std::vector<std::shared_ptr<ValueArray>>(col_count, nullptr));
      std::vector<std::vector<std::shared_ptr<ValueArray>>> column_offset_segs(
          cpu_count, std::vector<std::shared_ptr<ValueArray>>(col_count, nullptr));
      std::vector<std::vector<std::shared_ptr<std::vector<bool>>>> null_bitmap_segs(
          cpu_count, std::vector<std::shared_ptr<std::vector<bool>>>(col_count, nullptr));
      std::vector<std::vector<std::shared_ptr<std::vector<uint8_t>>>>
          offset_null_bitmap_segs(
              cpu_count,
              std::vector<std::shared_ptr<std::vector<uint8_t>>>(col_count, nullptr));
      const auto stride = (entry_count + cpu_count - 1) / cpu_count;
      for (size_t i = 0, start_entry = 0; start_entry < entry_count;
           ++i, start_entry += stride) {
        const auto end_entry = std::min(entry_count, start_entry + stride);
        child_threads.push_back(std::async(std::launch::async,
                                           fetch,
                                           std::ref(column_value_segs[i]),
                                           std::ref(column_offset_segs[i]),
                                           std::ref(null_bitmap_segs[i]),
                                           std::ref(offset_null_bitmap_segs[i]),
                                           non_lazy_cols,
                                           start_entry,
                                           end_entry));
      }
      for (auto& child : child_threads) {
        row_count += child.get();
      }
      {
        auto timer = DEBUG_TIMER("append rows to arrow");
        for (int i = 0; i < schema->num_fields(); ++i) {
          if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
            continue;
          }

          for (size_t j = 0; j < cpu_count; ++j) {
            if (!column_value_segs[j][i]) {
              continue;
            }
            append(builders[i],
                   *column_value_segs[j][i],
                   column_offset_segs[j][i],
                   null_bitmap_segs[j][i],
                   offset_null_bitmap_segs[j][i]);
          }
        }
      }
    } else {
      row_count = fetch(column_values,
                        column_offsets,
                        null_bitmaps,
                        offset_null_bitmaps,
                        non_lazy_cols,
                        size_t(0),
                        entry_count);
      {
        auto timer = DEBUG_TIMER("append rows to arrow single thread");
        for (int i = 0; i < schema->num_fields(); ++i) {
          if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
            continue;
          }

          append(builders[i],
                 *column_values[i],
                 column_offsets[i],
                 null_bitmaps[i],
                 offset_null_bitmaps[i]);
        }
      }
    }

    {
      auto timer = DEBUG_TIMER("finish builders");
      for (size_t i = 0; i < col_count; ++i) {
        if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
          continue;
        }

        result_columns[i] = finishColumnBuilder(builders[i]);
      }
    }
  }

  return ARROW_RECORDBATCH_MAKE(schema, row_count, result_columns);
}

std::shared_ptr<arrow::Table> ArrowResultSetConverter::getArrowTable(
    const std::shared_ptr<arrow::Schema>& schema) const {
  const auto col_count = results_->colCount();
  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns(col_count);

  if (results_->isEmpty()) {
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      result_columns[col_idx] = std::make_shared<arrow::ChunkedArray>(
          arrow::MakeArrayOfNull(schema->field(col_idx)->type(), 0).ValueOrDie());
    }
    return arrow::Table::Make(schema, result_columns, 0);
  }

  // Currently (20211203) there is no support for non-negative top_n_.
  CHECK_LT(top_n_, 0);

  const size_t entry_count = top_n_ < 0
                                 ? results_->entryCount()
                                 : std::min(size_t(top_n_), results_->entryCount());

  CHECK_GT(entry_count, 0);

  std::vector<ColumnBuilder> builders(col_count);
  for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
    initializeColumnBuilder(
        builders[col_idx], results_->colType(col_idx), col_idx, schema->field(col_idx));
  }

  bool columnar_conversion_possible =
      results_->isDirectColumnarConversionPossible() &&
      results_->getQueryMemDesc().getQueryDescriptionType() ==
          QueryDescriptionType::Projection;

  std::vector<bool> columnar_conversion_flags(col_count, false);
  const auto& lazy_fetch_info = results_->getLazyFetchInfo();
  if (columnar_conversion_possible) {
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      bool use_columnar_conversion = columnar_conversion_possible;

      // Lazily fetched column are not supportd by columnar converter.
      bool is_lazy =
          lazy_fetch_info.empty() ? false : lazy_fetch_info[col_idx].is_lazily_fetched;

      use_columnar_conversion = use_columnar_conversion && !is_lazy;

      // Some types are not supported by columnar converter.
      switch (builders[col_idx].physical_type->id()) {
        case hdk::ir::Type::kBoolean:
        case hdk::ir::Type::kTime:
        case hdk::ir::Type::kDate:
        case hdk::ir::Type::kTimestamp:
          use_columnar_conversion = false;
          break;
        default:
          break;
      }

      // Dictionaries are not supported by columnar converter.
      if (builders[col_idx].field->type()->id() == arrow::Type::DICTIONARY) {
        use_columnar_conversion = false;
      }

      columnar_conversion_flags[col_idx] = use_columnar_conversion;
    }
  }

  {
    auto timer = DEBUG_TIMER("columnar conversion");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, col_count),
                      [&](tbb::blocked_range<size_t> br) {
                        for (size_t col_idx = br.begin(); col_idx < br.end(); ++col_idx) {
                          if (columnar_conversion_flags[col_idx]) {
                            convert_column(builders[col_idx].physical_type,
                                           results_,
                                           col_idx,
                                           entry_count,
                                           result_columns[col_idx]);
                          }
                        }
                      });
  }

  bool use_rowwise_conversion = std::any_of(std::begin(columnar_conversion_flags),
                                            std::end(columnar_conversion_flags),
                                            [](bool val) { return val == false; });

  if (use_rowwise_conversion) {
    auto timer = DEBUG_TIMER("row-wise conversion");

    using ColumnValues = std::vector<std::shared_ptr<ValueArray>>;
    using NullBitmaps = std::vector<std::shared_ptr<std::vector<bool>>>;

    ColumnValues column_values(col_count, nullptr);
    NullBitmaps null_bitmaps(col_count, nullptr);

    size_t row_size_bytes = 0;
    for (size_t i = 0; i < col_count; i++) {
      row_size_bytes += results_->colType(i)->size();
    }
    CHECK_GT(row_size_bytes, 0);

    const size_t stride = std::clamp(entry_count / cpu_threads() / 2,
                                     65536 / row_size_bytes,
                                     8 * 1048576 / row_size_bytes);
    const size_t segments_count = (entry_count + stride - 1) / stride;

    std::vector<ColumnValues> column_value_segs(segments_count,
                                                ColumnValues(col_count, nullptr));
    std::vector<ColumnValues> column_offset_segs(segments_count,
                                                 ColumnValues(col_count, nullptr));
    std::vector<NullBitmaps> null_bitmap_segs(segments_count,
                                              NullBitmaps(col_count, nullptr));
    std::vector<std::vector<std::shared_ptr<std::vector<uint8_t>>>>
        offset_null_bitmap_segs(
            segments_count,
            std::vector<std::shared_ptr<std::vector<uint8_t>>>(col_count, nullptr));

    std::atomic<size_t> row_count = 0;

    if (results_->isTruncated()) {
      // TODO(dmitriim) This approach can be optimized for large offsets.
      // We'll process it without tbb for now, because it's hard to keep track of the
      // number of rows received across all threads.
      auto timer = DEBUG_TIMER("fetch data single thread with limit/offset");
      row_count += convert_rowwise(results_,
                                   builders,
                                   device_type_,
                                   column_value_segs[0],
                                   column_offset_segs[0],
                                   null_bitmap_segs[0],
                                   offset_null_bitmap_segs[0],
                                   columnar_conversion_flags,
                                   0,
                                   entry_count,
                                   results_->isTruncated());
    } else {
      auto timer = DEBUG_TIMER("fetch data in parallel_for");
      threading::parallel_for(
          static_cast<size_t>(0), entry_count, stride, [&](size_t start_entry) {
            const size_t i = start_entry / stride;
            const size_t end_entry = std::min(entry_count, start_entry + stride);
            row_count += convert_rowwise(results_,
                                         builders,
                                         device_type_,
                                         column_value_segs[i],
                                         column_offset_segs[i],
                                         null_bitmap_segs[i],
                                         offset_null_bitmap_segs[i],
                                         columnar_conversion_flags,
                                         start_entry,
                                         end_entry);
          });
    }

    CHECK_LE(row_count.load(), entry_count);

    {
      auto timer = DEBUG_TIMER("append rows to arrow, finish builders");
      threading::parallel_for(static_cast<size_t>(0), col_count, [&](size_t i) {
        if (!columnar_conversion_flags[i]) {
          for (size_t j = 0; j < segments_count; ++j) {
            if (column_value_segs[j][i]) {
              append(builders[i],
                     *column_value_segs[j][i],
                     column_offset_segs[j][i],
                     null_bitmap_segs[j][i],
                     offset_null_bitmap_segs[j][i]);
            }
          }
          result_columns[i] =
              std::make_shared<arrow::ChunkedArray>(finishColumnBuilder(builders[i]));
        }
      });
    }
  }

  return arrow::Table::Make(schema, result_columns);
}

namespace {

std::shared_ptr<arrow::DataType> get_arrow_type(const hdk::ir::Type* type,
                                                const ExecutorDeviceType device_type) {
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      return arrow::boolean();
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          return arrow::int8();
        case 2:
          return arrow::int16();
        case 4:
          return arrow::int32();
        case 8:
          return arrow::int64();
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return arrow::float32();
        case hdk::ir::FloatingPointType::kDouble:
          return arrow::float64();
        default:
          break;
      }
      break;
    case hdk::ir::Type::kExtDictionary: {
      auto value_type = std::make_shared<arrow::StringType>();
      return dictionary(arrow::int32(), value_type, false);
    }
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      return arrow::utf8();
    case hdk::ir::Type::kDecimal:
      // No reason to use 256-bit decimals since we always import 64-bit values.
      CHECK_EQ(type->size(), 8);
      return arrow::decimal(std::min(type->as<hdk::ir::DecimalType>()->precision(), 38),
                            type->as<hdk::ir::DecimalType>()->scale());
    case hdk::ir::Type::kTime:
      return time32(arrow::TimeUnit::SECOND);
    case hdk::ir::Type::kDate: {
      // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
      // Currently support for date32() is missing in cuDF.Hence, if client requests for
      // date on GPU, return date64() for the time being, till support is added.
      if (device_type == ExecutorDeviceType::GPU) {
        return arrow::date64();
      } else {
        return arrow::date32();
      }
    }
    case hdk::ir::Type::kTimestamp: {
      auto unit = type->as<hdk::ir::TimestampType>()->unit();
      switch (unit) {
        case hdk::ir::TimeUnit::kSecond:
          return timestamp(arrow::TimeUnit::SECOND);
        case hdk::ir::TimeUnit::kMilli:
          return timestamp(arrow::TimeUnit::MILLI);
        case hdk::ir::TimeUnit::kMicro:
          return timestamp(arrow::TimeUnit::MICRO);
        case hdk::ir::TimeUnit::kNano:
          return timestamp(arrow::TimeUnit::NANO);
        default:
          throw std::runtime_error(
              "Unsupported timestamp precision for Arrow result sets: " + toString(unit));
      }
    }
    case hdk::ir::Type::kVarLenArray: {
      auto elem_type = type->as<hdk::ir::VarLenArrayType>()->elemType();
      auto arrow_elem_type = get_arrow_type(elem_type, device_type);
      return arrow::list(arrow_elem_type);
    }
    default:
      break;
  }
  throw std::runtime_error(type->toString() + " is not supported in Arrow result sets.");
}

}  // namespace

std::shared_ptr<arrow::Field> ArrowResultSetConverter::makeField(
    const std::string name,
    const hdk::ir::Type* target_type) const {
  return arrow::field(
      name, get_arrow_type(target_type, device_type_), target_type->nullable());
}

std::shared_ptr<arrow::Schema> ArrowResultSetConverter::makeSchema() const {
  const auto col_count = results_->colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  CHECK(col_names_.empty() || col_names_.size() == col_count);
  for (size_t i = 0; i < col_count; ++i) {
    const auto type = results_->colType(i);
    fields.push_back(makeField(col_names_.empty() ? "" : col_names_[i], type));
  }
#if ARROW_CONVERTER_DEBUG
  VLOG(1) << "Arrow fields: ";
  for (const auto& f : fields) {
    VLOG(1) << "\t" << f->ToString(true);
  }
#endif
  return arrow::schema(fields);
}

void ArrowResultSet::deallocateArrowResultBuffer(const ArrowResult& result,
                                                 const ExecutorDeviceType device_type,
                                                 const size_t device_id) {
#ifndef _MSC_VER
  // CPU buffers skip the sm handle, serializing the entire RecordBatch to df.
  // Remove shared memory on sysmem
  if (!result.sm_handle.empty()) {
    CHECK_EQ(sizeof(key_t), result.sm_handle.size());
    const key_t& schema_key = *(key_t*)(&result.sm_handle[0]);
    auto shm_id = shmget(schema_key, result.sm_size, 0666);
    if (shm_id < 0) {
      throw std::runtime_error(
          "failed to get an valid shm ID w/ given shm key of the schema");
    }
    if (-1 == shmctl(shm_id, IPC_RMID, 0)) {
      throw std::runtime_error("failed to deallocate Arrow schema on errorno(" +
                               std::to_string(errno) + ")");
    }
  }

  if (device_type == ExecutorDeviceType::CPU) {
    CHECK_EQ(sizeof(key_t), result.df_handle.size());
    const key_t& df_key = *(key_t*)(&result.df_handle[0]);
    auto shm_id = shmget(df_key, result.df_size, 0666);
    if (shm_id < 0) {
      throw std::runtime_error(
          "failed to get an valid shm ID w/ given shm key of the data");
    }
    if (-1 == shmctl(shm_id, IPC_RMID, 0)) {
      throw std::runtime_error("failed to deallocate Arrow data frame");
    }
  }
  // CUDA buffers become owned by the caller, and will automatically be freed
  // TODO: What if the client never takes ownership of the result? we may want to
  // establish a check to see if the GPU buffer still exists, and then free it.
#endif
}

void ArrowResultSetConverter::initializeColumnBuilder(
    ColumnBuilder& column_builder,
    const hdk::ir::Type* col_type,
    const size_t results_col_slot_idx,
    const std::shared_ptr<arrow::Field>& field) const {
  column_builder.field = field;
  column_builder.col_type = col_type;
  column_builder.physical_type =
      col_type->isExtDictionary() ? col_type->ctx().integer(col_type->size()) : col_type;

  auto value_type = field->type();
  if (col_type->isExtDictionary()) {
    auto timer = DEBUG_TIMER("Translate string dictionary to Arrow dictionary");
    column_builder.builder.reset(new arrow::StringDictionary32Builder());
    // add values to the builder
    const int dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();

    // ResultSet::rowCount(), unlike ResultSet::entryCount(), will return
    // the actual number of rows in the result set, taking into account
    // things like any limit and offset set
    const size_t result_set_rows = results_->rowCount();
    // result_set_rows guaranteed > 0 by parent
    CHECK_GT(result_set_rows, 0UL);

    auto sdp = results_->getStringDictionaryProxy(dict_id);
    const size_t dictionary_proxy_entries = sdp->entryCount();
    const double dictionary_to_result_size_ratio =
        static_cast<double>(dictionary_proxy_entries) / result_set_rows;

    // We are conservative with when we do a bulk dictionary fetch,
    // even though it is generally more efficient than dictionary unique value "plucking",
    // for the following reasons:
    // 1) The number of actual distinct dictionary values can be much lower than the
    // number of result rows, but without getting the expression range (and that would
    // only work in some cases), we don't know by how much
    // 2) Regardless of the effect of #1, the size of the dictionary generated via
    // the "pluck" method will always be at worst equal in size, and very likely
    // significantly smaller, than the dictionary created by the bulk dictionary
    // fetch method, and smaller Arrow dictionaries are always a win when it comes to
    // sending the Arrow results over the wire, and for lowering the processing load
    // for clients (which often is a web browser with a lot less compute and memory
    // resources than our server.)

    const bool do_dictionary_bulk_fetch =
        result_set_rows > min_result_size_for_bulk_dictionary_fetch_ &&
        dictionary_to_result_size_ratio <=
            max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_;

    arrow::StringBuilder str_array_builder;

    if (do_dictionary_bulk_fetch) {
      VLOG(1) << "Arrow dictionary creation: bulk copying all dictionary "
              << " entries for column at offset " << results_col_slot_idx << ". "
              << "Column has " << dictionary_proxy_entries << " string entries"
              << " for a result set with " << result_set_rows << " rows.";
      column_builder.string_remap_mode =
          ArrowStringRemapMode::ONLY_TRANSIENT_STRINGS_REMAPPED;
      auto str_list = results_->getStringDictionaryPayloadCopy(dict_id);
      ARROW_THROW_NOT_OK(str_array_builder.AppendValues(str_list));

      // When we fetch the bulk dictionary, we need to also fetch
      // the transient entries only contained in the proxy.
      // These values are always negative (starting at -2), and so need
      // to be remapped to point to the corresponding entries in the Arrow
      // dictionary (they are placed at the end after the materialized
      // string entries from StringDictionary)

      int32_t crt_transient_id = static_cast<int32_t>(str_list.size());
      auto const& transient_vecmap = sdp->getTransientVector();
      for (unsigned index = 0; index < transient_vecmap.size(); ++index) {
        ARROW_THROW_NOT_OK(str_array_builder.Append(*transient_vecmap[index]));
        auto const old_id = StringDictionaryProxy::transientIndexToId(index);
        CHECK(column_builder.string_remapping
                  .insert(std::make_pair(old_id, crt_transient_id++))
                  .second);
      }
    } else {
      // Pluck unique dictionary values from ResultSet column
      VLOG(1) << "Arrow dictionary creation: serializing unique result set dictionary "
              << " entries for column at offset " << results_col_slot_idx << ". "
              << "Column has " << dictionary_proxy_entries << " string entries"
              << " for a result set with " << result_set_rows << " rows.";
      column_builder.string_remap_mode = ArrowStringRemapMode::ALL_STRINGS_REMAPPED;

      // ResultSet::getUniqueStringsForDictEncodedTargetCol returns a pair of two vectors,
      // the first of int32_t values containing the unique string ids found for
      // results_col_slot_idx in the result set, the second containing the associated
      // unique strings. Note that the unique string for a unique string id are both
      // placed at the same offset in their respective vectors

      auto unique_ids_and_strings =
          results_->getUniqueStringsForDictEncodedTargetCol(results_col_slot_idx);
      const auto& unique_ids = unique_ids_and_strings.first;
      const auto& unique_strings = unique_ids_and_strings.second;
      ARROW_THROW_NOT_OK(str_array_builder.AppendValues(unique_strings));
      const int32_t num_unique_strings = unique_strings.size();
      CHECK_EQ(num_unique_strings, unique_ids.size());
      // We need to remap ALL string id values given the Arrow dictionary
      // will have "holes", i.e. it is a sparse representation of the underlying
      // StringDictionary
      for (int32_t unique_string_idx = 0; unique_string_idx < num_unique_strings;
           ++unique_string_idx) {
        CHECK(
            column_builder.string_remapping
                .insert(std::make_pair(unique_ids[unique_string_idx], unique_string_idx))
                .second);
      }
      // Note we don't need to get transients from proxy as they are already handled in
      // ResultSet::getUniqueStringsForDictEncodedTargetCol
    }

    std::shared_ptr<arrow::StringArray> string_array;
    ARROW_THROW_NOT_OK(str_array_builder.Finish(&string_array));

    auto dict_builder =
        dynamic_cast<arrow::StringDictionary32Builder*>(column_builder.builder.get());
    CHECK(dict_builder);

    ARROW_THROW_NOT_OK(dict_builder->InsertMemoValues(*string_array));
  } else {
    ARROW_THROW_NOT_OK(arrow::MakeBuilder(
        arrow::default_memory_pool(), value_type, &column_builder.builder));
  }
}

std::shared_ptr<arrow::Array> ArrowResultSetConverter::finishColumnBuilder(
    ColumnBuilder& column_builder) const {
  std::shared_ptr<arrow::Array> values;
  ARROW_THROW_NOT_OK(column_builder.builder->Finish(&values));
  return values;
}

namespace {

template <typename BUILDER_TYPE, typename VALUE_ARRAY_TYPE>
void appendToColumnBuilder(ArrowResultSetConverter::ColumnBuilder& column_builder,
                           arrow::ArrayBuilder* arrow_builder,
                           const ValueArray& values,
                           const std::shared_ptr<std::vector<bool>>& is_valid) {
  static_assert(!std::is_same<BUILDER_TYPE, arrow::StringDictionary32Builder>::value,
                "Dictionary encoded string builder requires function specialization.");

  std::vector<VALUE_ARRAY_TYPE> vals = boost::get<std::vector<VALUE_ARRAY_TYPE>>(values);

  if (scale_epoch_values<BUILDER_TYPE>()) {
    auto scale_sec_to_millisec = [](auto seconds) { return seconds * kMilliSecsPerSec; };
    auto scale_values = [&](auto epoch) {
      return std::is_same<BUILDER_TYPE, arrow::Date32Builder>::value
                 ? DateConverters::get_epoch_days_from_seconds(epoch)
                 : scale_sec_to_millisec(epoch);
    };
    std::transform(vals.begin(), vals.end(), vals.begin(), scale_values);
  }

  auto typed_builder = dynamic_cast<BUILDER_TYPE*>(arrow_builder);
  CHECK(typed_builder);
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals, *is_valid));
  } else {
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals));
  }
}

template <>
void appendToColumnBuilder<arrow::Decimal128Builder, int64_t>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    arrow::ArrayBuilder* arrow_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  std::vector<int64_t> vals = boost::get<std::vector<int64_t>>(values);
  auto typed_builder = dynamic_cast<arrow::Decimal128Builder*>(arrow_builder);
  CHECK(typed_builder);
  CHECK_EQ(is_valid->size(), vals.size());
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    for (size_t i = 0; i < vals.size(); i++) {
      const auto v = vals[i];
      const auto valid = (*is_valid)[i];
      if (valid) {
        ARROW_THROW_NOT_OK(typed_builder->Append(v));
      } else {
        ARROW_THROW_NOT_OK(typed_builder->AppendNull());
      }
    }
  } else {
    for (const auto& v : vals) {
      ARROW_THROW_NOT_OK(typed_builder->Append(v));
    }
  }
}

template <>
void appendToColumnBuilder<arrow::StringDictionary32Builder, int32_t>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    arrow::ArrayBuilder* arrow_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  auto typed_builder = dynamic_cast<arrow::StringDictionary32Builder*>(arrow_builder);
  CHECK(typed_builder);

  std::vector<int32_t> vals = boost::get<std::vector<int32_t>>(values);
  // remap negative values if ArrowStringRemapMode == ONLY_TRANSIENT_STRINGS_REMAPPED or
  // everything if ALL_STRINGS_REMAPPED
  CHECK(column_builder.string_remap_mode != ArrowStringRemapMode::INVALID);
  for (size_t i = 0; i < vals.size(); i++) {
    auto& val = vals[i];
    if ((column_builder.string_remap_mode == ArrowStringRemapMode::ALL_STRINGS_REMAPPED ||
         val < 0) &&
        (!is_valid || (*is_valid)[i])) {
      vals[i] = column_builder.string_remapping.at(val);
    }
  }

  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    // TODO(adb): Generate this instead of the boolean bitmap
    std::vector<uint8_t> transformed_bitmap;
    transformed_bitmap.reserve(is_valid->size());
    std::for_each(
        is_valid->begin(), is_valid->end(), [&transformed_bitmap](const bool is_valid) {
          transformed_bitmap.push_back(is_valid ? 1 : 0);
        });

    ARROW_THROW_NOT_OK(typed_builder->AppendIndices(
        vals.data(), static_cast<int64_t>(vals.size()), transformed_bitmap.data()));
  } else {
    ARROW_THROW_NOT_OK(
        typed_builder->AppendIndices(vals.data(), static_cast<int64_t>(vals.size())));
  }
}

template <>
void appendToColumnBuilder<arrow::StringBuilder, std::string>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    arrow::ArrayBuilder* arrow_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  const std::vector<std::string>& vals = boost::get<std::vector<std::string>>(values);
  auto typed_builder = dynamic_cast<arrow::StringBuilder*>(arrow_builder);
  CHECK(typed_builder);
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    for (size_t i = 0; i < vals.size(); ++i) {
      if (is_valid->at(i)) {
        ARROW_THROW_NOT_OK(typed_builder->Append(vals[i]));
      } else {
        ARROW_THROW_NOT_OK(typed_builder->AppendNull());
      }
    }
  } else {
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals));
  }
}

}  // anonymous namespace

void ArrowResultSetConverter::append(
    ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<ValueArray>& offset_values,
    const std::shared_ptr<std::vector<bool>>& is_valid,
    const std::shared_ptr<std::vector<uint8_t>>& offset_is_valid) const {
  auto arrow_builder = column_builder.builder.get();
  auto elem_builder = arrow_builder;
  auto type = column_builder.physical_type;
  auto elem_type = type;
  if (elem_type->isVarLenArray()) {
    elem_type = elem_type->as<hdk::ir::VarLenArrayType>()->elemType();
    elem_builder = dynamic_cast<arrow::ListBuilder*>(elem_builder)->value_builder();
  }
  if (column_builder.col_type->isExtDictionary()) {
    CHECK(column_builder.physical_type
              ->isInt32());  // assume all dicts use none-encoded type for now
    appendToColumnBuilder<arrow::StringDictionary32Builder, int32_t>(
        column_builder, elem_builder, values, is_valid);
    return;
  }
  switch (elem_type->id()) {
    case hdk::ir::Type::kBoolean:
      appendToColumnBuilder<arrow::BooleanBuilder, bool>(
          column_builder, elem_builder, values, is_valid);
      break;
    case hdk::ir::Type::kInteger:
      switch (elem_type->size()) {
        case 1:
          appendToColumnBuilder<arrow::Int8Builder, int8_t>(
              column_builder, elem_builder, values, is_valid);
          break;
        case 2:
          appendToColumnBuilder<arrow::Int16Builder, int16_t>(
              column_builder, elem_builder, values, is_valid);
          break;
        case 4:
          appendToColumnBuilder<arrow::Int32Builder, int32_t>(
              column_builder, elem_builder, values, is_valid);
          break;
        case 8:
          appendToColumnBuilder<arrow::Int64Builder, int64_t>(
              column_builder, elem_builder, values, is_valid);
          break;
        default:
          throw std::runtime_error(column_builder.col_type->toString() +
                                   " is not supported in Arrow result sets.");
      }
      break;
    case hdk::ir::Type::kDecimal:
      appendToColumnBuilder<arrow::Decimal128Builder, int64_t>(
          column_builder, elem_builder, values, is_valid);
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (elem_type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          appendToColumnBuilder<arrow::FloatBuilder, float>(
              column_builder, elem_builder, values, is_valid);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          appendToColumnBuilder<arrow::DoubleBuilder, double>(
              column_builder, elem_builder, values, is_valid);
          break;
        default:
          throw std::runtime_error(column_builder.col_type->toString() +
                                   " is not supported in Arrow result sets.");
      }
      break;
    case hdk::ir::Type::kTime:
      appendToColumnBuilder<arrow::Time32Builder, int32_t>(
          column_builder, elem_builder, values, is_valid);
      break;
    case hdk::ir::Type::kTimestamp:
      appendToColumnBuilder<arrow::TimestampBuilder, int64_t>(
          column_builder, elem_builder, values, is_valid);
      break;
    case hdk::ir::Type::kDate:
      device_type_ == ExecutorDeviceType::GPU
          ? appendToColumnBuilder<arrow::Date64Builder, int64_t>(
                column_builder, elem_builder, values, is_valid)
          : appendToColumnBuilder<arrow::Date32Builder, int32_t>(
                column_builder, elem_builder, values, is_valid);
      break;
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      appendToColumnBuilder<arrow::StringBuilder, std::string>(
          column_builder, elem_builder, values, is_valid);
      break;
    default:
      // TODO(miyu): support more scalar types.
      throw std::runtime_error(column_builder.col_type->toString() +
                               " is not supported in Arrow result sets.");
  }
  // Append offset values
  if (type->isVarLenArray()) {
    CHECK(offset_values);
    auto& offsets = boost::get<std::vector<int32_t>>(*offset_values);
    auto* list_builder = dynamic_cast<arrow::ListBuilder*>(arrow_builder);
    if (offset_is_valid) {
      ARROW_THROW_NOT_OK(list_builder->AppendValues(
          offsets.data(), offsets.size() - 1, offset_is_valid->data()));
    } else {
      ARROW_THROW_NOT_OK(list_builder->AppendValues(offsets.data(), offsets.size() - 1));
    }
  }
}
