/*
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

#include "ArrowStorageUtils.h"

#include "IR/Context.h"
#include "Shared/InlineNullValues.h"

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include <immintrin.h>

#include <iostream>

using namespace std::string_literals;

namespace {

int int_pow(int val, int power) {
  int res = 1;
  for (int i = 0; i < power; ++i) {
    res *= val;
  }
  return res;
}

void convertBoolBitmapBufferWithNulls(int8_t* dst,
                                      const uint8_t* src,
                                      const uint8_t* bitmap,
                                      size_t offs,
                                      size_t length,
                                      int8_t null_value) {
  size_t start_full_byte = (offs + 7) / 8;
  size_t end_full_byte = (offs + length) / 8;
  size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
  size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
  size_t dst_offs = 0;

  for (size_t i = 0; i < head_bits; ++i) {
    auto is_null = (~bitmap[offs / 8] >> (offs % 8 + i)) & 1;
    auto val = (src[offs / 8] >> (offs % 8 + i)) & 1;
    dst[dst_offs++] = is_null ? null_value : val;
  }

  for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    auto inversed_bitmap = ~bitmap[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
      auto val = (source >> bitmap_offset) & 1;
      dst[dst_offs++] = is_null ? null_value : val;
    }
  }

  for (size_t i = 0; i < tail_bits; ++i) {
    auto is_null = (~bitmap[end_full_byte] >> i) & 1;
    auto val = (src[end_full_byte] >> i) & 1;
    dst[dst_offs++] = is_null ? null_value : val;
  }
}

// offs - an offset is source buffer it bits
// length - a number of bits to convert
void convertBoolBitmapBufferWithoutNulls(int8_t* dst,
                                         const uint8_t* src,
                                         size_t offs,
                                         size_t length) {
  size_t start_full_byte = (offs + 7) / 8;
  size_t end_full_byte = (offs + length) / 8;
  size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
  size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
  size_t dst_offs = 0;

  for (size_t i = 0; i < head_bits; ++i) {
    dst[dst_offs++] = (src[offs / 8] >> (offs % 8 + i)) & 1;
  }

  for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      dst[dst_offs++] = (source >> bitmap_offset) & 1;
    }
  }

  for (size_t i = 0; i < tail_bits; ++i) {
    dst[dst_offs++] = (src[end_full_byte] >> i) & 1;
  }
}

template <typename T>
void copyArrayDataReplacingNulls(T* dst,
                                 std::shared_ptr<arrow::Array> arr,
                                 size_t offs,
                                 size_t length) {
  auto null_value = inline_null_value<T>();
  if (arr->null_count() == arr->length()) {
    if constexpr (std::is_same_v<T, bool>) {
      std::fill(reinterpret_cast<int8_t*>(dst),
                reinterpret_cast<int8_t*>(dst + length),
                null_value);
    } else {
      std::fill(dst, dst + length, null_value);
    }
    return;
  }

  auto src = reinterpret_cast<const T*>(arr->data()->buffers[1]->data());
  if (arr->null_count() == 0) {
    if constexpr (std::is_same_v<T, bool>) {
      convertBoolBitmapBufferWithoutNulls(reinterpret_cast<int8_t*>(dst),
                                          reinterpret_cast<const uint8_t*>(src),
                                          offs,
                                          length);
    } else {
      std::copy(src + offs, src + offs + length, dst);
    }
    return;
  }

  const uint8_t* bitmap_data = arr->null_bitmap_data();
  if constexpr (std::is_same_v<T, bool>) {
    convertBoolBitmapBufferWithNulls(reinterpret_cast<int8_t*>(dst),
                                     reinterpret_cast<const uint8_t*>(src),
                                     bitmap_data,
                                     offs,
                                     length,
                                     null_value);
    return;
  }

  size_t start_full_byte = (offs + 7) / 8;
  size_t end_full_byte = (offs + length) / 8;
  size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
  size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
  size_t dst_offs = 0;
  size_t src_offs = offs;

  for (size_t i = 0; i < head_bits; ++i) {
    auto is_null = (~bitmap_data[offs / 8] >> (offs % 8 + i)) & 1;
    auto val = src[src_offs++];
    dst[dst_offs++] = is_null ? null_value : val;
  }

  for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte; ++bitmap_idx) {
    auto inversed_bitmap = ~bitmap_data[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
      auto val = src[src_offs++];
      dst[dst_offs++] = is_null ? null_value : val;
    }
  }

  for (size_t i = 0; i < tail_bits; ++i) {
    auto is_null = (~bitmap_data[end_full_byte] >> i) & 1;
    auto val = src[src_offs++];
    dst[dst_offs++] = is_null ? null_value : val;
  }
}

template <typename T>
void copyArrayDataReplacingNulls(T* dst, std::shared_ptr<arrow::Array> arr) {
  copyArrayDataReplacingNulls(dst, arr, 0, arr->length());
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    bool force_copy) {
  if (!force_copy && !std::is_same_v<T, bool> && arr->null_count() == 0) {
    // for boolean columns we still need to convert bitmaps to array
    return arr;
  }

  auto resultBuf = arrow::AllocateBuffer(sizeof(T) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<T*>(resultBuf->mutable_data());

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        size_t offset = 0;
                        for (int i = 0; i < c; i++) {
                          offset += arr->chunk(i)->length();
                        }
                        copyArrayDataReplacingNulls<T>(resultData + offset,
                                                       arr->chunk(c));
                      }
                    });

  std::shared_ptr<arrow::Array> array;
  if constexpr (std::is_same_v<T, bool>) {
    array = std::make_shared<arrow::Int8Array>(arr->length(), std::move(resultBuf));
  } else if (arr->type()->id() == arrow::Type::NA) {
    using ResultArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using ArrayType = typename arrow::TypeTraits<ResultArrowType>::ArrayType;
    array = std::make_shared<ArrayType>(arr->length(), std::move(resultBuf));
  } else {
    array = std::make_shared<arrow::PrimitiveArray>(
        arr->type(), arr->length(), std::move(resultBuf));
  }

  return std::make_shared<arrow::ChunkedArray>(array);
}

template <typename ArrowIntType, typename ResultIntType>
void copyDateReplacingNulls(ResultIntType* dst, std::shared_ptr<arrow::Array> arr) {
  auto src = reinterpret_cast<const ArrowIntType*>(arr->data()->buffers[1]->data());
  auto null_value = inline_null_value<ResultIntType>();
  auto length = arr->length();

  if (arr->null_count() == length) {
    std::fill(dst, dst + length, null_value);
  } else if (arr->null_count() == 0) {
    std::transform(src, src + length, dst, [](ArrowIntType v) {
      return static_cast<ResultIntType>(v);
    });
  } else {
    const uint8_t* bitmap_data = arr->null_bitmap_data();

    size_t end_full_byte = length / 8;
    size_t tail_bits = length % 8;
    size_t offs = 0;

    for (size_t bitmap_idx = 0; bitmap_idx < end_full_byte; ++bitmap_idx) {
      auto inversed_bitmap = ~bitmap_data[bitmap_idx];
      for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
        auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
        auto val = static_cast<ArrowIntType>(src[offs]);
        dst[offs] = is_null ? null_value : val;
        ++offs;
      }
    }

    for (size_t i = 0; i < tail_bits; ++i) {
      auto is_null = (~bitmap_data[end_full_byte] >> i) & 1;
      auto val = static_cast<ArrowIntType>(src[offs]);
      dst[offs] = is_null ? null_value : val;
      ++offs;
    }
  }
}

template <typename ArrowIntType, typename ResultIntType>
std::shared_ptr<arrow::ChunkedArray> convertDateReplacingNulls(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  auto resultBuf =
      arrow::AllocateBuffer(sizeof(ResultIntType) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<ResultIntType*>(resultBuf->mutable_data());

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        size_t offset = 0;
                        for (int i = 0; i < c; i++) {
                          offset += arr->chunk(i)->length();
                        }
                        copyDateReplacingNulls<ArrowIntType, ResultIntType>(
                            resultData + offset, arr->chunk(c));
                      }
                    });

  using ResultArrowType = typename arrow::CTypeTraits<ResultIntType>::ArrowType;
  using ArrayType = typename arrow::TypeTraits<ResultArrowType>::ArrayType;

  auto array = std::make_shared<ArrayType>(arr->length(), std::move(resultBuf));
  return std::make_shared<arrow::ChunkedArray>(array);
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  size_t elems_count = 0;
  int32_t arrays_count = 0;
  std::vector<int32_t> out_elem_offsets;
  std::vector<int32_t> out_offset_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  out_offset_offsets.reserve(arr->num_chunks());

  for (auto& chunk : arr->chunks()) {
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    CHECK(chunk_list);

    out_offset_offsets.push_back(arrays_count);
    out_elem_offsets.push_back(static_cast<int32_t>(elems_count));

    auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
    arrays_count += chunk->length();
    elems_count += offset_data[chunk->length()] - offset_data[0];
  }

  if (elems_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("Input arrow array is too big for conversion.");
  }

  auto offset_buf =
      arrow::AllocateBuffer(sizeof(int32_t) * arr->length() + 1).ValueOrDie();
  auto offset_ptr = reinterpret_cast<int32_t*>(offset_buf->mutable_data());

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * elems_count).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto elem_offset = out_elem_offsets[c];
          auto offset_offset = out_offset_offsets[c];
          auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array = chunk_list->values();

          auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
          if (chunk_list->null_count() == 0) {
            auto first_elem_offset = offset_data[0];
            auto elems_to_copy = offset_data[chunk_list->length()] - first_elem_offset;
            copyArrayDataReplacingNulls<T>(
                elem_ptr + elem_offset, elem_array, first_elem_offset, elems_to_copy);
            std::transform(offset_data,
                           offset_data + chunk_list->length(),
                           offset_ptr + offset_offset,
                           [offs = elem_offset - first_elem_offset](uint32_t val) {
                             return (val + offs) * sizeof(T);
                           });
          } else {
            bool use_negative_offset = false;
            for (int64_t i = 0; i < chunk_list->length(); ++i) {
              offset_ptr[offset_offset++] = use_negative_offset ? -elem_offset * sizeof(T)
                                                                : elem_offset * sizeof(T);
              if (chunk_list->IsNull(i)) {
                use_negative_offset = true;
              } else {
                use_negative_offset = false;
                auto elems_to_copy = offset_data[i + 1] - offset_data[i];
                copyArrayDataReplacingNulls<T>(
                    elem_ptr + elem_offset, elem_array, offset_data[i], elems_to_copy);
                elem_offset += elems_to_copy;
              }
            }
          }
        }
      });
  auto last_chunk = arr->chunk(arr->num_chunks() - 1);
  offset_ptr[arr->length()] = static_cast<int32_t>(
      last_chunk->IsNull(last_chunk->length() - 1) ? -elems_count * sizeof(T)
                                                   : elems_count * sizeof(T));

  std::shared_ptr<arrow::Array> elem_array;
  auto list_type = arr->type();
  if constexpr (std::is_same_v<T, bool>) {
    elem_array = std::make_shared<arrow::Int8Array>(elems_count, std::move(elem_buf));
    list_type = arrow::list(arrow::int8());
  } else {
    using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;
    elem_array = std::make_shared<ElemsArrayType>(elems_count, std::move(elem_buf));
  }

  auto list_array = std::make_shared<arrow::ListArray>(
      list_type, arr->length(), std::move(offset_buf), elem_array);
  return std::make_shared<arrow::ChunkedArray>(list_array);
}

template <typename T, bool float_conversion>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenDecimalArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int scale = 1) {
  size_t elems_count = 0;
  int32_t arrays_count = 0;
  std::vector<int32_t> out_elem_offsets;
  std::vector<int32_t> out_offset_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  out_offset_offsets.reserve(arr->num_chunks());

  for (auto& chunk : arr->chunks()) {
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    CHECK(chunk_list);

    out_offset_offsets.push_back(arrays_count);
    out_elem_offsets.push_back(static_cast<int32_t>(elems_count));

    auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
    arrays_count += chunk->length();
    elems_count += offset_data[chunk->length()] - offset_data[0];
  }

  if (elems_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("Input arrow array is too big for conversion.");
  }

  auto offset_buf =
      arrow::AllocateBuffer(sizeof(int32_t) * arr->length() + 1).ValueOrDie();
  auto offset_ptr = reinterpret_cast<int32_t*>(offset_buf->mutable_data());

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * elems_count).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto elem_offset = out_elem_offsets[c];
          auto offset_offset = out_offset_offsets[c];
          auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array = chunk_list->values();
          auto decimalArray =
              std::dynamic_pointer_cast<arrow::Decimal128Array>(elem_array);
          auto floatArray = std::dynamic_pointer_cast<arrow::DoubleArray>(elem_array);
          auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);

          bool use_negative_offset = false;
          for (int64_t i = 0; i < chunk_list->length(); ++i) {
            offset_ptr[offset_offset++] =
                use_negative_offset ? -elem_offset * sizeof(T) : elem_offset * sizeof(T);
            if (chunk_list->IsNull(i)) {
              use_negative_offset = true;
            } else {
              use_negative_offset = false;
              auto offs = offset_data[i];
              auto len = offset_data[i + 1] - offset_data[i];
              for (uint32_t j = 0; j < len; ++j) {
                if (elem_array->IsNull(offs + j)) {
                  elem_ptr[elem_offset + j] = inline_null_value<T>();
                } else if constexpr (float_conversion) {
                  double val = floatArray->Value(offs + j);
                  elem_ptr[elem_offset + j] = static_cast<int64_t>(val * scale);
                } else {
                  arrow::Decimal128 val(decimalArray->GetValue(offs + j));
                  elem_ptr[elem_offset + j] = static_cast<int64_t>(val);
                }
              }
              elem_offset += len;
            }
          }
        }
      });
  auto last_chunk = arr->chunk(arr->num_chunks() - 1);
  offset_ptr[arr->length()] = static_cast<int32_t>(
      last_chunk->IsNull(last_chunk->length() - 1) ? -elems_count * sizeof(T)
                                                   : elems_count * sizeof(T));

  using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
  using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;

  auto elem_array = std::make_shared<ElemsArrayType>(elems_count, std::move(elem_buf));
  auto list_array =
      std::make_shared<arrow::ListArray>(arrow::list(std::make_shared<ElemsArrowType>()),
                                         arr->length(),
                                         std::move(offset_buf),
                                         elem_array);
  return std::make_shared<arrow::ChunkedArray>(list_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenStringArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    StringDictionary* dict) {
  size_t elems_count = 0;
  int32_t arrays_count = 0;
  std::vector<int32_t> out_elem_offsets;
  std::vector<int32_t> out_offset_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  out_offset_offsets.reserve(arr->num_chunks());

  for (auto& chunk : arr->chunks()) {
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    CHECK(chunk_list);

    out_offset_offsets.push_back(arrays_count);
    out_elem_offsets.push_back(static_cast<int32_t>(elems_count));

    auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
    arrays_count += chunk->length();
    elems_count += offset_data[chunk->length()] - offset_data[0];
  }

  if (elems_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("Input arrow array is too big for conversion.");
  }

  auto offset_buf =
      arrow::AllocateBuffer(sizeof(int32_t) * arr->length() + 1).ValueOrDie();
  auto offset_ptr = reinterpret_cast<int32_t*>(offset_buf->mutable_data());

  auto elem_buf = arrow::AllocateBuffer(sizeof(int32_t) * elems_count).ValueOrDie();
  auto elem_ptr = reinterpret_cast<int32_t*>(elem_buf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto elem_offset = out_elem_offsets[c];
          auto offset_offset = out_offset_offsets[c];
          auto chunk_array = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array =
              std::dynamic_pointer_cast<arrow::StringArray>(chunk_array->values());
          CHECK(elem_array);

          bool use_negative_offset = false;
          for (int64_t i = 0; i < chunk_array->length(); ++i) {
            offset_ptr[offset_offset++] = use_negative_offset
                                              ? -elem_offset * sizeof(int32_t)
                                              : elem_offset * sizeof(int32_t);
            if (chunk_array->IsNull(i)) {
              use_negative_offset = true;
            } else {
              use_negative_offset = false;
              auto offs = chunk_array->value_offset(i);
              auto len = chunk_array->value_length(i);
              for (int j = 0; j < len; ++j) {
                auto view = elem_array->GetView(offs + j);
                elem_ptr[elem_offset++] =
                    dict->getOrAdd(std::string_view(view.data(), view.length()));
              }
            }
          }
        }
      });
  auto last_chunk = arr->chunk(arr->num_chunks() - 1);
  offset_ptr[arr->length()] = static_cast<int32_t>(
      last_chunk->IsNull(last_chunk->length() - 1) ? -elems_count * sizeof(int32_t)
                                                   : elems_count * sizeof(int32_t));

  auto elem_array = std::make_shared<arrow::Int32Array>(elems_count, std::move(elem_buf));
  auto list_array = std::make_shared<arrow::ListArray>(
      arrow::list(arrow::int32()), arr->length(), std::move(offset_buf), elem_array);
  return std::make_shared<arrow::ChunkedArray>(list_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenArray(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type,
    StringDictionary* dict) {
  auto list_type = std::dynamic_pointer_cast<arrow::ListType>(arr->type());
  if (!list_type) {
    throw std::runtime_error("Unsupported large Arrow list:: "s +
                             arr->type()->ToString());
  }

  auto elem_type = type->as<hdk::ir::ArrayBaseType>()->elemType();
  if (elem_type->isInteger() || elem_type->isDateTime()) {
    switch (elem_type->size()) {
      case 1:
        return replaceNullValuesVarlenArrayImpl<int8_t>(arr);
      case 2:
        return replaceNullValuesVarlenArrayImpl<int16_t>(arr);
      case 4:
        return replaceNullValuesVarlenArrayImpl<int32_t>(arr);
      case 8:
        return replaceNullValuesVarlenArrayImpl<int64_t>(arr);
      default:
        throw std::runtime_error("Unsupported integer or datetime array: "s +
                                 type->toString());
    }
  } else if (elem_type->isFloatingPoint()) {
    switch (elem_type->size()) {
      case 4:
        return replaceNullValuesVarlenArrayImpl<float>(arr);
      case 8:
        return replaceNullValuesVarlenArrayImpl<double>(arr);
    }
  } else if (elem_type->isBoolean()) {
    return replaceNullValuesVarlenArrayImpl<bool>(arr);
  } else if (elem_type->isExtDictionary()) {
    CHECK_EQ(elem_type->size(), 4);
    return replaceNullValuesVarlenStringArrayImpl(arr, dict);
  } else if (elem_type->isDecimal()) {
    // Due to JSON parser limitation in Arrow 5.0 we might need to convert
    // float64 do decimal.
    bool float_conversion = list_type->value_type()->id() == arrow::Type::DOUBLE;
    auto dec_type = elem_type->as<hdk::ir::DecimalType>();
    switch (elem_type->size()) {
      case 8:
        if (float_conversion) {
          return replaceNullValuesVarlenDecimalArrayImpl<int64_t, true>(
              arr, int_pow(10, dec_type->scale()));
        } else {
          return replaceNullValuesVarlenDecimalArrayImpl<int64_t, false>(arr);
        }
    }
  }

  throw std::runtime_error("Unsupported varlen array: "s + type->toString());
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int list_size) {
  int64_t total_length = 0;
  std::vector<size_t> out_elem_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  for (auto& chunk : arr->chunks()) {
    out_elem_offsets.push_back(total_length);
    total_length += chunk->length() * list_size;
  }

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * total_length).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  auto null_array_value = inline_null_array_value<T>();
  auto null_value = inline_null_value<T>();

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        auto chunk_array =
                            std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
                        auto elem_array = chunk_array->values();

                        auto dst_ptr = elem_ptr + out_elem_offsets[c];
                        for (int64_t i = 0; i < chunk_array->length(); ++i) {
                          if (chunk_array->IsNull(i)) {
                            dst_ptr[0] = null_array_value;
                            for (int j = 1; j < list_size; ++j) {
                              dst_ptr[j] = null_value;
                            }
                          } else {
                            // We add NULL elements if input array is too short and cut
                            // too long arrays.
                            auto offs = chunk_array->value_offset(i);
                            auto len = std::min(chunk_array->value_length(i), list_size);
                            copyArrayDataReplacingNulls(dst_ptr, elem_array, offs, len);
                            for (int j = len; j < list_size; ++j) {
                              dst_ptr[j] = null_value;
                            }
                          }
                          dst_ptr += list_size;
                        }
                      }
                    });

  std::shared_ptr<arrow::Array> elem_array;
  if constexpr (std::is_same_v<T, bool>) {
    elem_array = std::make_shared<arrow::Int8Array>(total_length, std::move(elem_buf));
  } else {
    using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;
    elem_array = std::make_shared<ElemsArrayType>(total_length, std::move(elem_buf));
  }

  return std::make_shared<arrow::ChunkedArray>(elem_array);
}

template <typename T, bool float_conversion>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeDecimalArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int list_size,
    int scale = 1) {
  int64_t total_length = 0;
  std::vector<size_t> out_elem_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  for (auto& chunk : arr->chunks()) {
    out_elem_offsets.push_back(total_length);
    total_length += chunk->length() * list_size;
  }

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * total_length).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  auto null_array_value = inline_null_array_value<T>();
  auto null_value = inline_null_value<T>();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto chunk_array = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array = chunk_array->values();
          auto decimalArray =
              std::dynamic_pointer_cast<arrow::Decimal128Array>(elem_array);
          auto floatArray = std::dynamic_pointer_cast<arrow::DoubleArray>(elem_array);

          auto dst_ptr = elem_ptr + out_elem_offsets[c];
          for (int64_t i = 0; i < chunk_array->length(); ++i) {
            if (chunk_array->IsNull(i)) {
              dst_ptr[0] = null_array_value;
              for (int j = 1; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            } else {
              // We add NULL elements if input array is too short and cut
              // too long arrays.
              auto offs = chunk_array->value_offset(i);
              auto len = std::min(chunk_array->value_length(i), list_size);
              for (int j = 0; j < len; ++j) {
                if (elem_array->IsNull(offs + j)) {
                  dst_ptr[j] = inline_null_value<T>();
                } else if constexpr (float_conversion) {
                  double val = floatArray->Value(offs + j);
                  dst_ptr[j] = static_cast<int64_t>(val * scale);
                } else {
                  arrow::Decimal128 val(decimalArray->GetValue(offs + j));
                  dst_ptr[j] = static_cast<int64_t>(val);
                }
              }
              for (int j = len; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            }
            dst_ptr += list_size;
          }
        }
      });

  using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
  using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;

  auto elem_array = std::make_shared<ElemsArrayType>(total_length, std::move(elem_buf));
  return std::make_shared<arrow::ChunkedArray>(elem_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeStringArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int list_size,
    StringDictionary* dict) {
  int64_t total_length = 0;
  std::vector<size_t> out_elem_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  for (auto& chunk : arr->chunks()) {
    out_elem_offsets.push_back(total_length);
    total_length += chunk->length() * list_size;
  }

  auto list_type = std::dynamic_pointer_cast<arrow::ListType>(arr->type());
  CHECK(list_type);
  auto elem_type = list_type->value_type();
  if (elem_type->id() != arrow::Type::STRING) {
    throw std::runtime_error(
        "Dictionary encoded string arrays are not supported in Arrow import: "s +
        list_type->ToString());
  }

  auto elem_buf = arrow::AllocateBuffer(sizeof(int32_t) * total_length).ValueOrDie();
  auto elem_ptr = reinterpret_cast<int32_t*>(elem_buf->mutable_data());

  auto null_array_value = inline_null_array_value<int32_t>();
  auto null_value = inline_null_value<int32_t>();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto chunk_array = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array =
              std::dynamic_pointer_cast<arrow::StringArray>(chunk_array->values());
          CHECK(elem_array);

          auto dst_ptr = elem_ptr + out_elem_offsets[c];
          for (int64_t i = 0; i < chunk_array->length(); ++i) {
            if (chunk_array->IsNull(i)) {
              dst_ptr[0] = null_array_value;
              for (int j = 1; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            } else {
              // We add NULL elements if input array is too short and cut
              // too long arrays.
              auto offs = chunk_array->value_offset(i);
              auto len = std::min(chunk_array->value_length(i), list_size);
              for (int j = 0; j < len; ++j) {
                auto view = elem_array->GetView(offs + j);
                dst_ptr[j] = dict->getOrAdd(std::string_view(view.data(), view.length()));
              }
              for (int j = len; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            }
            dst_ptr += list_size;
          }
        }
      });

  using ElemsArrowType = typename arrow::CTypeTraits<int32_t>::ArrowType;
  using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;

  auto elem_array = std::make_shared<ElemsArrayType>(total_length, std::move(elem_buf));
  return std::make_shared<arrow::ChunkedArray>(elem_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeArray(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type,
    StringDictionary* dict) {
  auto elem_type = type->as<hdk::ir::FixedLenArrayType>()->elemType();
  int list_size = type->as<hdk::ir::FixedLenArrayType>()->numElems();
  if (elem_type->isInteger() || elem_type->isDateTime()) {
    switch (elem_type->size()) {
      case 1:
        return replaceNullValuesFixedSizeArrayImpl<int8_t>(arr, list_size);
      case 2:
        return replaceNullValuesFixedSizeArrayImpl<int16_t>(arr, list_size);
      case 4:
        return replaceNullValuesFixedSizeArrayImpl<int32_t>(arr, list_size);
      case 8:
        return replaceNullValuesFixedSizeArrayImpl<int64_t>(arr, list_size);
      default:
        throw std::runtime_error("Unsupported integer or datetime array: "s +
                                 type->toString());
    }
  } else if (elem_type->isFloatingPoint()) {
    switch (elem_type->size()) {
      case 4:
        return replaceNullValuesFixedSizeArrayImpl<float>(arr, list_size);
      case 8:
        return replaceNullValuesFixedSizeArrayImpl<double>(arr, list_size);
    }
  } else if (elem_type->isBoolean()) {
    return replaceNullValuesFixedSizeArrayImpl<bool>(arr, list_size);
  } else if (elem_type->isExtDictionary()) {
    CHECK_EQ(elem_type->size(), 4);
    return replaceNullValuesFixedSizeStringArrayImpl(arr, list_size, dict);
  } else if (elem_type->isDecimal()) {
    // Due to JSON parser limitation in Arrow 5.0 we might need to convert
    // float64 do decimal.
    auto list_type = std::dynamic_pointer_cast<arrow::ListType>(arr->type());
    bool float_conversion = list_type->value_type()->id() == arrow::Type::DOUBLE;
    auto dec_type = elem_type->as<hdk::ir::DecimalType>();
    switch (elem_type->size()) {
      case 8:
        if (float_conversion) {
          return replaceNullValuesFixedSizeDecimalArrayImpl<int64_t, true>(
              arr, list_size, int_pow(10, dec_type->scale()));
        } else {
          return replaceNullValuesFixedSizeDecimalArrayImpl<int64_t, false>(arr,
                                                                            list_size);
        }
    }
  }

  throw std::runtime_error("Unsupported fixed size array: "s + type->toString());
}

template <typename IntType, typename ChunkType>
std::shared_ptr<arrow::ChunkedArray> convertDecimalToInteger(
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  size_t column_size = 0;
  std::vector<int> offsets(arr_col_chunked_array->num_chunks());
  for (int i = 0; i < arr_col_chunked_array->num_chunks(); i++) {
    offsets[i] = column_size;
    column_size += arr_col_chunked_array->chunk(i)->length();
  }

  std::shared_ptr<arrow::Buffer> result_buffer;
  auto res = arrow::AllocateBuffer(column_size * sizeof(IntType));
  CHECK(res.ok());
  result_buffer = std::move(res).ValueOrDie();

  IntType* buffer_data = reinterpret_cast<IntType*>(result_buffer->mutable_data());
  tbb::parallel_for(
      tbb::blocked_range(0, arr_col_chunked_array->num_chunks()),
      [buffer_data, &offsets, arr_col_chunked_array](auto& range) {
        for (int chunk_idx = range.begin(); chunk_idx < range.end(); chunk_idx++) {
          auto offset = offsets[chunk_idx];
          IntType* chunk_buffer = buffer_data + offset;

          auto decimalArray = std::static_pointer_cast<arrow::Decimal128Array>(
              arr_col_chunked_array->chunk(chunk_idx));
          auto empty =
              arr_col_chunked_array->null_count() == arr_col_chunked_array->length();
          for (int i = 0; i < decimalArray->length(); i++) {
            if (empty || decimalArray->null_count() == decimalArray->length() ||
                decimalArray->IsNull(i)) {
              chunk_buffer[i] = inline_null_value<IntType>();
            } else {
              arrow::Decimal128 val(decimalArray->GetValue(i));
              chunk_buffer[i] =
                  static_cast<int64_t>(val);  // arrow can cast only to int64_t
            }
          }
        }
      });
  auto array = std::make_shared<ChunkType>(column_size, result_buffer);
  return std::make_shared<arrow::ChunkedArray>(array);
}

}  // anonymous namespace

std::shared_ptr<arrow::ChunkedArray> replaceNullValues(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type,
    StringDictionary* dict,
    bool force_single_chunk) {
  bool force_copy = force_single_chunk && (arr->chunks().size() > 1);
  if (type->isTime()) {
    if (type->size() != 8) {
      throw std::runtime_error("Unsupported time type for Arrow import: "s +
                               type->toString());
    }
    return convertDateReplacingNulls<int32_t, int64_t>(arr);
  }
  if (type->isDate()) {
    if (type->as<hdk::ir::DateTimeBaseType>()->unit() != hdk::ir::TimeUnit::kDay) {
      throw std::runtime_error("Unsupported date type for Arrow import: "s +
                               type->toString());
    }

    switch (type->size()) {
      case 2:
        return convertDateReplacingNulls<int32_t, int16_t>(arr);
      case 4:
        return replaceNullValuesImpl<int32_t>(arr, force_copy);
      case 8:
        return convertDateReplacingNulls<int32_t, int64_t>(arr);
      default:
        throw std::runtime_error("Unsupported date type for Arrow import: "s +
                                 type->toString());
    }
  } else if (type->isInteger() || type->isTimestamp()) {
    switch (type->size()) {
      case 1:
        return replaceNullValuesImpl<int8_t>(arr, force_copy);
      case 2:
        return replaceNullValuesImpl<int16_t>(arr, force_copy);
      case 4:
        return replaceNullValuesImpl<int32_t>(arr, force_copy);
      case 8:
        return replaceNullValuesImpl<int64_t>(arr, force_copy);
      default:
        throw std::runtime_error("Unsupported integer/datetime type for Arrow import: "s +
                                 type->toString());
    }
  } else if (type->isFloatingPoint()) {
    switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
      case hdk::ir::FloatingPointType::kFloat:
        return replaceNullValuesImpl<float>(arr, force_copy);
      case hdk::ir::FloatingPointType::kDouble:
        return replaceNullValuesImpl<double>(arr, force_copy);
    }
  } else if (type->isBoolean()) {
    return replaceNullValuesImpl<bool>(arr, force_copy);
  } else if (type->isFixedLenArray()) {
    return replaceNullValuesFixedSizeArray(arr, type, dict);
  } else if (type->isVarLenArray()) {
    return replaceNullValuesVarlenArray(arr, type, dict);
  }
  throw std::runtime_error("Unexpected type for Arrow import: "s + type->toString());
}

std::shared_ptr<arrow::ChunkedArray> convertDecimalToInteger(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type) {
  CHECK(type->isDecimal());
  switch (type->size()) {
    case 2:
      return convertDecimalToInteger<int16_t, arrow::Int16Array>(arr);
    case 4:
      return convertDecimalToInteger<int32_t, arrow::Int32Array>(arr);
    case 8:
      return convertDecimalToInteger<int64_t, arrow::Int64Array>(arr);
    default:
      // TODO: throw unsupported decimal type exception
      throw std::runtime_error("Unsupported decimal type: " + type->toString());
  }
}

const hdk::ir::Type* getTargetImportType(hdk::ir::Context& ctx,
                                         const arrow::DataType& type) {
  using namespace arrow;
  switch (type.id()) {
    case Type::INT8:
      return ctx.int8();
    case Type::INT16:
      return ctx.int16();
    case Type::INT32:
      return ctx.int32();
    case Type::INT64:
      return ctx.int64();
    case Type::BOOL:
      return ctx.boolean();
    case Type::FLOAT:
      return ctx.fp32();
    case Type::DATE32:
      return ctx.date32();
    case Type::DATE64:
      return ctx.timestamp(hdk::ir::TimeUnit::kMilli);
    case Type::DOUBLE:
      return ctx.fp64();
    case Type::STRING:
      return ctx.extDict(ctx.text(), 0);
    case Type::NA:
      return ctx.fp64();
    case arrow::Type::DICTIONARY: {
      const auto& dict_type = static_cast<const arrow::DictionaryType&>(type);
      if (dict_type.value_type()->id() == Type::STRING) {
        return ctx.extDict(ctx.text(), 0);
      }
    } break;
    case Type::DECIMAL: {
      const auto& decimal_type = static_cast<const arrow::DecimalType&>(type);
      return ctx.decimal64(decimal_type.precision(), decimal_type.scale());
    }
    case Type::TIME32:
      if (static_cast<const arrow::Time32Type&>(type).unit() == arrow::TimeUnit::SECOND) {
        return ctx.time64(hdk::ir::TimeUnit::kSecond);
      }
      break;
    case Type::TIMESTAMP:
      switch (static_cast<const arrow::TimestampType&>(type).unit()) {
        case TimeUnit::SECOND:
          return ctx.timestamp(hdk::ir::TimeUnit::kSecond);
        case TimeUnit::MILLI:
          return ctx.timestamp(hdk::ir::TimeUnit::kMilli);
        case TimeUnit::MICRO:
          return ctx.timestamp(hdk::ir::TimeUnit::kMicro);
        case TimeUnit::NANO:
          return ctx.timestamp(hdk::ir::TimeUnit::kNano);
        default:
          break;
      }
      break;
    default:
      break;
  }
  throw std::runtime_error(type.ToString() + " is not yet supported. id: " + type.name());
}

std::shared_ptr<arrow::DataType> getArrowImportType(hdk::ir::Context& ctx,
                                                    const hdk::ir::Type* type) {
  using namespace arrow;
  switch (type->id()) {
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          return int8();
        case 2:
          return int16();
        case 4:
          return int32();
        case 8:
          return int64();
        default:
          break;
      }
      break;
    case hdk::ir::Type::kBoolean:
      return arrow::boolean();
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return float32();
        case hdk::ir::FloatingPointType::kDouble:
          return float64();
        default:
          break;
      }
      break;
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
    case hdk::ir::Type::kExtDictionary:
      return utf8();
    case hdk::ir::Type::kDecimal: {
      auto dec_type = type->as<hdk::ir::DecimalType>();
      return decimal(dec_type->precision(), dec_type->scale());
    }
    case hdk::ir::Type::kTime:
      return time32(TimeUnit::SECOND);
    case hdk::ir::Type::kDate:
      return arrow::date32();
    case hdk::ir::Type::kTimestamp:
      switch (type->as<hdk::ir::DateTimeBaseType>()->unit()) {
        case hdk::ir::TimeUnit::kSecond:
          return timestamp(TimeUnit::SECOND);
        case hdk::ir::TimeUnit::kMilli:
          return timestamp(TimeUnit::MILLI);
        case hdk::ir::TimeUnit::kMicro:
          return timestamp(TimeUnit::MICRO);
        case hdk::ir::TimeUnit::kNano:
          return timestamp(TimeUnit::NANO);
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray: {
      auto elem_type = type->as<hdk::ir::ArrayBaseType>()->elemType();
      if (elem_type->isDecimal()) {
        // Arrow 5.0 JSON parser doesn't support decimals. Use float64 instead
        // and do the conversion on import. Due to precision problems imported
        // data might not fully match the source data.
        // TODO: change it after moving to Arrow 6.0
        return list(float64());
      } else {
        // Arrow JSON parser doesn't support conversion to fixed size lists.
        // So we use variable length lists in Arrow in all cases and then do
        // the conversion.
        return list(getArrowImportType(ctx, elem_type));
      }
    }
    default:
      break;
  }
  throw std::runtime_error(type->toString() + " is not supported in Arrow import.");
}

namespace {

template <typename IndexType>
std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr) {
  // calculate offsets for every fragment in bulk
  size_t bulk_size = 0;
  std::vector<int> offsets(arr->num_chunks());
  for (int i = 0; i < arr->num_chunks(); i++) {
    offsets[i] = bulk_size;
    bulk_size += arr->chunk(i)->length();
  }

  std::vector<std::string_view> bulk(bulk_size);

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&bulk, &arr, &offsets](const tbb::blocked_range<int>& r) {
                      for (int i = r.begin(); i < r.end(); i++) {
                        auto chunk =
                            std::static_pointer_cast<arrow::StringArray>(arr->chunk(i));
                        auto offset = offsets[i];
                        for (int j = 0; j < chunk->length(); j++) {
                          auto view = chunk->GetView(j);
                          bulk[offset + j] = std::string_view(view.data(), view.length());
                        }
                      }
                    });

  std::shared_ptr<arrow::Buffer> indices_buf;
  auto res = arrow::AllocateBuffer(bulk_size * sizeof(int32_t));
  CHECK(res.ok());
  indices_buf = std::move(res).ValueOrDie();
  auto raw_data = reinterpret_cast<int*>(indices_buf->mutable_data());
  dict->getOrAddBulk(bulk, raw_data);

  if constexpr (std::is_same_v<IndexType, uint32_t>) {
    auto array = std::make_shared<arrow::Int32Array>(bulk_size, indices_buf);
    return std::make_shared<arrow::ChunkedArray>(array);
  } else {
    // We have to convert to a narrower index type. Indexes which don't fit
    // the type are replaced with invalid id.
    static_assert(sizeof(IndexType) < sizeof(int32_t));
    static_assert(std::is_unsigned_v<IndexType>);
    auto converted_res = arrow::AllocateBuffer(bulk_size * sizeof(IndexType));
    CHECK(converted_res.ok());
    std::shared_ptr<arrow::Buffer> converted_indices_buf =
        std::move(converted_res).ValueOrDie();
    auto raw_converted_data =
        reinterpret_cast<IndexType*>(converted_indices_buf->mutable_data());
    for (size_t i = 0; i < bulk_size; ++i) {
      if (raw_data[i] > static_cast<int32_t>(std::numeric_limits<IndexType>::max()) ||
          raw_data[i] == inline_null_value<int>()) {
        raw_converted_data[i] = static_cast<IndexType>(StringDictionary::INVALID_STR_ID);
      } else {
        raw_converted_data[i] = static_cast<IndexType>(raw_data[i]);
      }
    }

    using IndexArrowType = typename arrow::CTypeTraits<IndexType>::ArrowType;
    using ArrayType = typename arrow::TypeTraits<IndexArrowType>::ArrayType;

    auto array = std::make_shared<ArrayType>(bulk_size, converted_indices_buf);
    return std::make_shared<arrow::ChunkedArray>(array);
  }
}

}  // anonymous namespace

std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type) {
  switch (type->size()) {
    case 4:
      return createDictionaryEncodedColumn<uint32_t>(dict, arr);
    case 2:
      return createDictionaryEncodedColumn<uint16_t>(dict, arr);
    case 1:
      return createDictionaryEncodedColumn<uint8_t>(dict, arr);
    default:
      throw std::runtime_error(
          "Unsupported OmniSci dictionary for Arrow strings import: "s +
          type->toString());
  }
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> convertArrowDictionary(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const hdk::ir::Type* type) {
  if (!type->isExtDictionary() || type->size() != 4) {
    throw std::runtime_error("Unsupported HDK dictionary for Arrow dictionary import: "s +
                             type->toString());
  }
  // TODO: allocate one big array and split it by fragments as it is done in
  // createDictionaryEncodedColumn
  std::vector<std::shared_ptr<arrow::Array>> converted_chunks;
  for (auto& chunk : arr->chunks()) {
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
    auto values = std::static_pointer_cast<arrow::StringArray>(dict_array->dictionary());
    std::vector<std::string_view> strings(values->length());
    for (int i = 0; i < values->length(); i++) {
      auto view = values->GetView(i);
      strings[i] = std::string_view(view.data(), view.length());
    }
    auto arrow_indices =
        std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
    std::vector<int> indices_mapping(values->length());
    dict->getOrAddBulk(strings, indices_mapping.data());

    // create new arrow chunk with remapped indices
    std::shared_ptr<arrow::Buffer> dict_indices_buf;
    auto res = arrow::AllocateBuffer(arrow_indices->length() * sizeof(int32_t));
    CHECK(res.ok());
    dict_indices_buf = std::move(res).ValueOrDie();
    auto raw_data = reinterpret_cast<int32_t*>(dict_indices_buf->mutable_data());

    for (int i = 0; i < arrow_indices->length(); i++) {
      raw_data[i] = indices_mapping[arrow_indices->Value(i)];
    }

    converted_chunks.push_back(
        std::make_shared<arrow::Int32Array>(arrow_indices->length(), dict_indices_buf));
  }
  return std::make_shared<arrow::ChunkedArray>(converted_chunks);
}

#if 1  // ndef _MSC_VER
__attribute__((target("avx512f"), optimize("no-tree-vectorize"))) void
encodeStrDictIndicesImpl(int16_t* encoded_indices_buf,
                         const int32_t* indices_buf,
                         const size_t num_elems) {
  VLOG(2) << "Running vectorized 16-bit conversion";
  constexpr int vector_window_bytes = (512 / 8);
  constexpr int vector_window =
      vector_window_bytes / sizeof(int32_t);  // elements in a vector
  const size_t encoded_buf_size_bytes = num_elems * sizeof(int16_t);
  std::memset(encoded_indices_buf, std::numeric_limits<uint16_t>::max(), num_elems);

  __m512i null_vals =
      _mm512_set1_epi32(static_cast<int>(std::numeric_limits<int32_t>::min()));
  __m512i out_of_range_vals =
      _mm512_set1_epi32(static_cast<int>(std::numeric_limits<uint16_t>::max()));

  const int vec_buf_end = floor(num_elems / vector_window) * vector_window;
  int pos = 0;
  int16_t* crt_encoded_indices_buf_ptr = encoded_indices_buf;
  const int32_t* crt_indices_buf_ptr = indices_buf;
  while (pos < vec_buf_end) {
    __m512i vec_buf = _mm512_load_epi32(crt_indices_buf_ptr);

    // first, replace null sentinels
    __mmask16 null_sentinel_mask = _mm512_cmpeq_epu32_mask(vec_buf, null_vals);

    // then, replace out of range elements with the corresponding null sentinel
    __mmask16 out_of_range_mask = _mm512_cmpgt_epu32_mask(vec_buf, out_of_range_vals);

    // union the masks
    __mmask16 nulls_mask = _mm512_knot(_mm512_kor(null_sentinel_mask, out_of_range_mask));

    // finally, convert all elements to signed int16, skipping masked elements which will
    // be left null
    _mm512_mask_cvtusepi32_storeu_epi16(crt_encoded_indices_buf_ptr, nulls_mask, vec_buf);

    crt_encoded_indices_buf_ptr += vector_window;
    crt_indices_buf_ptr += vector_window;
    pos += vector_window;
  }

  // remainder
  for (int i = pos; i < int(num_elems); i++) {
    encoded_indices_buf[i] = indices_buf[i] == std::numeric_limits<int32_t>::min() ||
                                     indices_buf[i] > std::numeric_limits<uint16_t>::max()
                                 ? std::numeric_limits<uint16_t>::max()
                                 : indices_buf[i];
  }
}

__attribute__((target("avx512f"))) void encodeStrDictIndicesImpl(
    int8_t* encoded_indices_buf,
    const int32_t* indices_buf,
    const size_t num_elems) {
  VLOG(2) << "Running vectorized 8-bit conversion";
  constexpr int vector_window_bytes = (512 / 8);
  constexpr int vector_window =
      vector_window_bytes / sizeof(int32_t);  // elements in a vector
  const size_t encoded_buf_size_bytes = num_elems * sizeof(int8_t);
  std::memset(encoded_indices_buf, std::numeric_limits<uint8_t>::max(), num_elems);

  __m512i null_vals =
      _mm512_set1_epi32(static_cast<int>(std::numeric_limits<int32_t>::min()));
  __m512i out_of_range_vals =
      _mm512_set1_epi32(static_cast<int>(std::numeric_limits<uint8_t>::max()));

  const int vec_buf_end = floor(num_elems / vector_window) * vector_window;
  int pos = 0;
  int8_t* crt_encoded_indices_buf_ptr = encoded_indices_buf;
  const int32_t* crt_indices_buf_ptr = indices_buf;
  while (pos < vec_buf_end) {
    __m512i vec_buf = _mm512_load_epi32(crt_indices_buf_ptr);

    // first, replace null sentinels
    __mmask16 null_sentinel_mask = _mm512_cmpneq_epu32_mask(vec_buf, null_vals);

    // then, replace out of range elements with the corresponding null sentinel
    __mmask16 nulls_mask =
        _mm512_mask_cmple_epu32_mask(null_sentinel_mask, vec_buf, out_of_range_vals);

    // union the masks
    // __mmask16 nulls_mask = _mm512_knot(_mm512_kor(null_sentinel_mask,
    // out_of_range_mask));

    // finally, convert all elements to signed int16, skipping masked elements which will
    // be left null
    _mm512_mask_cvtusepi32_storeu_epi8(crt_encoded_indices_buf_ptr, nulls_mask, vec_buf);

    crt_encoded_indices_buf_ptr += vector_window;
    crt_indices_buf_ptr += vector_window;
    pos += vector_window;
  }

  // remainder
  for (int i = pos; i < int(num_elems); i++) {
    encoded_indices_buf[i] = indices_buf[i] == std::numeric_limits<int32_t>::min() ||
                                     indices_buf[i] > std::numeric_limits<uint8_t>::max()
                                 ? std::numeric_limits<uint8_t>::max()
                                 : indices_buf[i];
  }
}

#endif

// TODO: centralize
#if defined(_MSC_VER)
#define DEFAULT_TARGET_ATTRIBUTE
#else
#define DEFAULT_TARGET_ATTRIBUTE __attribute__((target("default")))
#endif

DEFAULT_TARGET_ATTRIBUTE void encodeStrDictIndicesImpl(int16_t* encoded_indices_buf,
                                                       const int32_t* indices_buf,
                                                       const size_t num_elems) {
  for (size_t i = 0; i < num_elems; i++) {
    encoded_indices_buf[i] = indices_buf[i] == std::numeric_limits<int32_t>::min() ||
                                     indices_buf[i] > std::numeric_limits<uint16_t>::max()
                                 ? std::numeric_limits<uint16_t>::max()
                                 : indices_buf[i];
  }
}

DEFAULT_TARGET_ATTRIBUTE void encodeStrDictIndicesImpl(int8_t* encoded_indices_buf,
                                                       const int32_t* indices_buf,
                                                       const size_t num_elems) {
  for (size_t i = 0; i < num_elems; i++) {
    encoded_indices_buf[i] = indices_buf[i] == std::numeric_limits<int32_t>::min() ||
                                     indices_buf[i] > std::numeric_limits<uint8_t>::max()
                                 ? std::numeric_limits<uint8_t>::max()
                                 : indices_buf[i];
  }
}

// dispatch the appropriate impl function depending on whether avx512 is available

void encodeStrDictIndices(int8_t* encoded_indices_buf,
                          const int32_t* indices_buf,
                          const size_t num_elems) {
  return encodeStrDictIndicesImpl(encoded_indices_buf, indices_buf, num_elems);
}

void encodeStrDictIndices(int16_t* encoded_indices_buf,
                          const int32_t* indices_buf,
                          const size_t num_elems) {
  return encodeStrDictIndicesImpl(encoded_indices_buf, indices_buf, num_elems);
}
