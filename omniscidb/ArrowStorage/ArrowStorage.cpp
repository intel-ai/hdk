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

#include "ArrowStorage.h"
#include "ArrowStorageUtils.h"

#include "IR/Type.h"
#include "Shared/ArrowUtil.h"
#include "Shared/measure.h"
#include "Shared/threading.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <arrow/csv/reader.h>
#include <arrow/io/api.h>
#include <arrow/json/reader.h>
#include <arrow/util/decimal.h>
#include <arrow/util/value_parsing.h>
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>

#pragma GCC diagnostic pop

using namespace std::string_literals;

namespace {

size_t computeTotalStringsLength(std::shared_ptr<arrow::ChunkedArray> arr,
                                 size_t offset,
                                 size_t rows) {
  size_t start_offset = offset;
  size_t chunk_no = 0;
  while (static_cast<size_t>(arr->chunk(chunk_no)->length()) <= start_offset) {
    start_offset -= arr->chunk(chunk_no)->length();
    ++chunk_no;
  }

  size_t rows_remain = rows;
  size_t total_bytes = 0;
  while (rows_remain) {
    auto chunk = arr->chunk(chunk_no);
    size_t rows_in_chunk = std::min(rows_remain, chunk->length() - start_offset);
    const int32_t* offsets = chunk->data()->GetValues<int32_t>(1);
    total_bytes +=
        std::abs(offsets[start_offset + rows_in_chunk]) - std::abs(offsets[start_offset]);
    rows_remain -= rows_in_chunk;
    start_offset = 0;
    ++chunk_no;
  }

  return total_bytes;
}

}  // anonymous namespace

arrow::Status ArenaMemoryPool::Allocate(int64_t size, uint8_t** out) {
  if (size < 0) {
    return arrow::Status::Invalid("negative malloc size");
  }
  if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
    return arrow::Status::OutOfMemory("malloc size overflows size_t");
  }
  *out = reinterpret_cast<uint8_t*>(arena.allocate(size));
  if (!(*out)) {
    return arrow::Status::UnknownError("failed to allocate memory in arena");
  }
  bytes_allocated_ += size;
  return arrow::Status::OK();
}

arrow::Status ArenaMemoryPool::Reallocate(int64_t old_size,
                                          int64_t new_size,
                                          uint8_t** ptr) {
  if (new_size < old_size) {
    (*ptr)[new_size] = '\0';
    return arrow::Status::OK();
  }
  if (old_size == new_size) {
    // noop
    return arrow::Status::OK();
  }
  uint8_t* new_ptr{nullptr};
  RETURN_NOT_OK(Allocate(new_size, &new_ptr));
  CHECK(new_ptr);
  std::memcpy(new_ptr, *ptr, static_cast<size_t>(std::min(new_size, old_size)));
  // we don't free, because the arena doesn't free
  *ptr = new_ptr;
  return arrow::Status::OK();
}

void ArenaMemoryPool::Free(uint8_t* buffer, int64_t size) {
  // noop
  return;
}

void ArrowStorage::fetchBuffer(const ChunkKey& key,
                               Data_Namespace::AbstractBuffer* dest,
                               const size_t num_bytes) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  size_t col_idx = static_cast<size_t>(key[CHUNK_KEY_COLUMN_IDX] - 1);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  CHECK_LT(col_idx, table.col_data.size());

  auto col_type =
      getColumnInfo(
          key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX], key[CHUNK_KEY_COLUMN_IDX])
          ->type;
  dest->reserve(num_bytes);
  if (!col_type->isVarLen()) {
    CHECK_EQ(key.size(), (size_t)4);
    size_t elem_size = col_type->size();
    fetchFixedLenData(table, frag_idx, col_idx, dest, num_bytes, elem_size);
  } else {
    CHECK_EQ(key.size(), (size_t)5);
    if (key[CHUNK_KEY_VARLEN_IDX] == 1) {
      if (!dest->hasEncoder()) {
        dest->initEncoder(col_type);
      }
      if (col_type->isString()) {
        fetchVarLenData(table, frag_idx, col_idx, dest, num_bytes);
      } else {
        CHECK(col_type->isVarLenArray());
        fetchVarLenArrayData(table,
                             frag_idx,
                             col_idx,
                             dest,
                             col_type->as<hdk::ir::ArrayBaseType>()->elemType()->size(),
                             num_bytes);
      }
    } else {
      CHECK_EQ(key[CHUNK_KEY_VARLEN_IDX], 2);
      fetchVarLenOffsets(table, frag_idx, col_idx, dest, num_bytes);
    }
  }
  dest->setSize(num_bytes);
}

std::unique_ptr<AbstractDataToken> ArrowStorage::getZeroCopyBufferMemory(
    const ChunkKey& key,
    size_t num_bytes) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  auto col_type =
      getColumnInfo(
          key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX], key[CHUNK_KEY_COLUMN_IDX])
          ->type;

  if (!col_type->isVarLen()) {
    size_t col_idx = static_cast<size_t>(key[CHUNK_KEY_COLUMN_IDX] - 1);
    size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
    CHECK_EQ(key.size(), (size_t)4);
    size_t elem_size = col_type->size();
    auto& frag = table.fragments[frag_idx];
    size_t rows_to_fetch = num_bytes ? num_bytes / elem_size : frag.row_count;
    const auto* fixed_type =
        dynamic_cast<const arrow::FixedWidthType*>(table.col_data[col_idx]->type().get());
    CHECK(fixed_type);
    size_t arrow_elem_size = fixed_type->bit_width() / 8;
    // For fixed size arrays we simply use elem type in arrow and therefore have to scale
    // to get a proper slice.
    size_t elems = elem_size / arrow_elem_size;
    CHECK_GT(elems, (size_t)0);
    auto data_to_fetch =
        table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset * elems),
                                       static_cast<int64_t>(rows_to_fetch * elems));
    if (data_to_fetch->num_chunks() == 1) {
      auto chunk = data_to_fetch->chunk(0);
      const int8_t* ptr =
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * arrow_elem_size);
      size_t chunk_size = chunk->length() * arrow_elem_size;
      return std::make_unique<ArrowChunkDataToken>(std::move(chunk), ptr, chunk_size);
    }
  }

  return nullptr;
}

void ArrowStorage::fetchFixedLenData(const TableData& table,
                                     size_t frag_idx,
                                     size_t col_idx,
                                     Data_Namespace::AbstractBuffer* dest,
                                     size_t num_bytes,
                                     size_t elem_size) const {
  auto& frag = table.fragments[frag_idx];
  size_t rows_to_fetch = num_bytes ? num_bytes / elem_size : frag.row_count;
  const auto* fixed_type =
      dynamic_cast<const arrow::FixedWidthType*>(table.col_data[col_idx]->type().get());
  CHECK(fixed_type);
  size_t arrow_elem_size = fixed_type->bit_width() / 8;
  // For fixed size arrays we simply use elem type in arrow and therefore have to scale
  // to get a proper slice.
  size_t elems = elem_size / arrow_elem_size;
  CHECK_GT(elems, (size_t)0);
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset * elems),
                                     static_cast<int64_t>(rows_to_fetch * elems));
  int8_t* dst_ptr = dest->getMemoryPtr();
  for (auto& chunk : data_to_fetch->chunks()) {
    size_t chunk_size = chunk->length() * arrow_elem_size;
    const int8_t* src_ptr =
        chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * arrow_elem_size);
    memcpy(dst_ptr, src_ptr, chunk_size);
    dst_ptr += chunk_size;
  }
}

void ArrowStorage::fetchVarLenOffsets(const TableData& table,
                                      size_t frag_idx,
                                      size_t col_idx,
                                      Data_Namespace::AbstractBuffer* dest,
                                      size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  CHECK_EQ(num_bytes, (frag.row_count + 1) * sizeof(uint32_t));
  // Number of fetched offsets is 1 greater than number of fetched rows.
  size_t rows_to_fetch = num_bytes ? num_bytes / sizeof(uint32_t) - 1 : frag.row_count;
  auto data_to_fetch = table.col_data[col_idx]->Slice(
      static_cast<int64_t>(frag.offset), static_cast<int64_t>(rows_to_fetch));
  uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dest->getMemoryPtr());
  uint32_t delta = 0;
  for (auto& chunk : data_to_fetch->chunks()) {
    const uint32_t* src_ptr = chunk->data()->GetValues<uint32_t>(1);
    delta -= *src_ptr;
    dst_ptr = std::transform(
        src_ptr, src_ptr + chunk->length(), dst_ptr, [delta](uint32_t val) {
          return val + delta;
        });
    delta += src_ptr[chunk->length()];
  }
  *dst_ptr = delta;
}

void ArrowStorage::fetchVarLenData(const TableData& table,
                                   size_t frag_idx,
                                   size_t col_idx,
                                   Data_Namespace::AbstractBuffer* dest,
                                   size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset), frag.row_count);
  int8_t* dst_ptr = dest->getMemoryPtr();
  size_t remained = num_bytes;
  for (auto& chunk : data_to_fetch->chunks()) {
    if (remained == 0) {
      break;
    }

    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    size_t chunk_size = offsets[chunk->length()] - offsets[0];
    chunk_size = std::min(chunk_size, num_bytes);
    memcpy(dst_ptr, chunk->data()->GetValues<int8_t>(2, offsets[0]), chunk_size);
    remained -= chunk_size;
    dst_ptr += chunk_size;
  }
}

void ArrowStorage::fetchVarLenArrayData(const TableData& table,
                                        size_t frag_idx,
                                        size_t col_idx,
                                        Data_Namespace::AbstractBuffer* dest,
                                        size_t elem_size,
                                        size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset), frag.row_count);
  int8_t* dst_ptr = dest->getMemoryPtr();
  size_t remained = num_bytes;
  for (auto& chunk : data_to_fetch->chunks()) {
    if (remained == 0) {
      break;
    }

    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    size_t chunk_size = offsets[chunk->length()] - offsets[0];
    chunk_size = std::min(chunk_size, num_bytes);
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    auto elem_array = chunk_list->values();
    memcpy(dst_ptr, elem_array->data()->GetValues<int8_t>(1, offsets[0]), chunk_size);
    remained -= chunk_size;
    dst_ptr += chunk_size;
  }
}

TableFragmentsInfo ArrowStorage::getTableMetadata(int db_id, int table_id) const {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(db_id, db_id_);
  CHECK_EQ(tables_.count(table_id), (size_t)1);
  auto& table = *tables_.at(table_id);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  if (table.fragments.empty()) {
    return getEmptyTableMetadata(table_id);
  }

  TableFragmentsInfo res;
  res.setPhysicalNumTuples(table.row_count);
  for (size_t frag_idx = 0; frag_idx < table.fragments.size(); ++frag_idx) {
    auto& frag = table.fragments[frag_idx];
    auto& frag_info = res.fragments.emplace_back();
    frag_info.fragmentId = static_cast<int>(frag_idx + 1);
    frag_info.physicalTableId = table_id;
    frag_info.setPhysicalNumTuples(frag.row_count);
    frag_info.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
    for (size_t col_idx = 0; col_idx < frag.metadata.size(); ++col_idx) {
      frag_info.setChunkMetadata(static_cast<int>(col_idx + 1), frag.metadata[col_idx]);
    }
  }
  return res;
}

TableFragmentsInfo ArrowStorage::getEmptyTableMetadata(int table_id) const {
  TableFragmentsInfo res;
  res.setPhysicalNumTuples(0);

  // Executor requires dummy empty fragment for empty tables
  FragmentInfo& empty_frag = res.fragments.emplace_back();
  empty_frag.fragmentId = 0;
  empty_frag.shadowNumTuples = 0;
  empty_frag.setPhysicalNumTuples(0);
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
  empty_frag.physicalTableId = table_id;
  res.fragments.push_back(empty_frag);

  return res;
}

const DictDescriptor* ArrowStorage::getDictMetadata(int dict_id, bool /*load_dict*/) {
  mapd_shared_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  CHECK_EQ(getSchemaId(dict_id), schema_id_);
  if (dicts_.count(dict_id)) {
    return dicts_.at(dict_id).get();
  }
  return nullptr;
}

TableInfoPtr ArrowStorage::createTable(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) {
  TableInfoPtr res;
  int table_id;
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);
  {
    mapd_unique_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
    mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);
    table_id = next_table_id_++;
    checkNewTableParams(table_name, columns, options);
    res = addTableInfo(
        db_id_, table_id, table_name, false, Data_Namespace::MemoryLevel::CPU_LEVEL, 0);
    int next_col_id = 1;
    std::unordered_map<int, int> dict_ids;
    for (auto& col : columns) {
      auto type = col.type;
      auto elem_type =
          type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
      // Positive dictionary id means we use existing dictionary. Other values
      // mean we have to create new dictionaries. Columns with equal negative
      // dict ids will share dictionaries.
      if (elem_type->isExtDictionary()) {
        auto dict_type = elem_type->as<hdk::ir::ExtDictionaryType>();
        auto sharing_id = dict_type->dictId();
        if (sharing_id < 0 && dict_ids.count(sharing_id)) {
          elem_type = ctx_.extDict(
              dict_type->elemType(), dict_ids.at(sharing_id), dict_type->size());
        } else if (sharing_id <= 0) {
          if (next_dict_id_ > MAX_DB_ID) {
            throw std::runtime_error("Dictionary count limit exceeded.");
          }

          int dict_id = addSchemaIdChecked(next_dict_id_++, schema_id_);
          auto dict_desc = std::make_unique<DictDescriptor>(
              db_id_, dict_id, col.name, 32, true, 1, table_name, true);
          dict_desc->stringDict = std::make_shared<StringDictionary>(
              DictRef{db_id_, dict_id}, table_name + "."s + col.name, true, false);
          if (sharing_id < 0) {
            dict_ids.emplace(sharing_id, dict_id);
          }
          dicts_.emplace(dict_id, std::move(dict_desc));
          elem_type = ctx_.extDict(dict_type->elemType(), dict_id, dict_type->size());
        }

        if (type->isFixedLenArray()) {
          type = ctx_.arrayFixed(type->as<hdk::ir::FixedLenArrayType>()->numElems(),
                                 elem_type,
                                 type->nullable());
        } else if (type->isVarLenArray()) {
          type = ctx_.arrayVarLen(elem_type,
                                  type->as<hdk::ir::VarLenArrayType>()->offsetSize(),
                                  type->nullable());
        } else {
          type = elem_type;
        }
      }
      auto col_info =
          addColumnInfo(db_id_, table_id, next_col_id++, col.name, type, false);
    }
    addRowidColumn(db_id_, table_id);
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    auto& name = columns[i].name;
    auto& type = columns[i].type;
    auto field = arrow::field(name, getArrowImportType(ctx_, type), type->nullable());
    fields.push_back(field);
  }
  auto schema = arrow::schema(fields);

  {
    auto [iter, inserted] = tables_.emplace(table_id, std::make_unique<TableData>());
    CHECK(inserted);
    auto& table = *iter->second;
    table.fragment_size = options.fragment_size;
    table.schema = schema;
  }

  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const std::vector<ColumnDescription>& columns,
                                            const TableOptions& options) {
  auto res = createTable(table_name, columns, options);
  appendArrowTable(at, table_name);
  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const TableOptions& options) {
  std::vector<ColumnDescription> columns;
  for (auto& field : at->schema()->fields()) {
    ColumnDescription desc{field->name(), getTargetImportType(ctx_, *field->type())};
    columns.emplace_back(std::move(desc));
  }
  return importArrowTable(at, table_name, columns, options);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at,
                                    const std::string& table_name) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendArrowTable(at, tinfo->table_id);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at, int table_id) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  if (!tables_.count(table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto& table = *tables_.at(table_id);
  compareSchemas(table.schema, at->schema());

  mapd_unique_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  std::vector<std::shared_ptr<arrow::ChunkedArray>> col_data;
  col_data.resize(at->columns().size());

  std::vector<DataFragment> fragments;
  // Compute size of the fragment. If the last existing fragment is not full, then it will
  // be merged with the first new fragment.
  size_t first_frag_size =
      std::min(table.fragment_size, static_cast<size_t>(at->num_rows()));
  if (!table.fragments.empty()) {
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      first_frag_size =
          std::min(first_frag_size, table.fragment_size - last_frag.row_count);
    }
  }
  // Now we can compute number of fragments to create.
  size_t frag_count =
      (static_cast<size_t>(at->num_rows()) + table.fragment_size - 1 - first_frag_size) /
          table.fragment_size +
      1;
  fragments.resize(frag_count);
  for (auto& frag : fragments) {
    frag.metadata.resize(at->columns().size());
  }

  mapd_shared_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  threading::parallel_for(
      threading::blocked_range(0, (int)at->columns().size()), [&](auto range) {
        for (auto col_idx = range.begin(); col_idx != range.end(); col_idx++) {
          auto col_info = getColumnInfo(db_id_, table_id, col_idx + 1);
          auto col_type = col_info->type;
          auto col_arr = at->column(col_idx);
          StringDictionary* dict = nullptr;
          auto elem_type =
              col_type->isArray()
                  ? dynamic_cast<const hdk::ir::ArrayBaseType*>(col_type)->elemType()
                  : col_type;
          if (elem_type->isExtDictionary()) {
            dict = dicts_
                       .at(dynamic_cast<const hdk::ir::ExtDictionaryType*>(elem_type)
                               ->dictId())
                       ->stringDict.get();
          }

          if (col_type->isDecimal()) {
            col_arr = convertDecimalToInteger(col_arr, col_type);
          } else if (col_type->isExtDictionary()) {
            switch (col_arr->type()->id()) {
              case arrow::Type::STRING:
                col_arr = createDictionaryEncodedColumn(dict, col_arr, col_type);
                break;
              case arrow::Type::DICTIONARY:
                col_arr = convertArrowDictionary(dict, col_arr, col_type);
                break;
              default:
                CHECK(false);
            }
          } else if (col_type->isString()) {
          } else {
            col_arr = replaceNullValues(col_arr, col_type, dict);
          }

          col_data[col_idx] = col_arr;

          bool compute_stats = !col_type->isString();
          if (compute_stats) {
            size_t elems_count = 1;
            if (col_type->isFixedLenArray()) {
              elems_count = col_type->size() / elem_type->size();
            }
            // Compute stats for each fragment.
            threading::parallel_for(
                threading::blocked_range(size_t(0), frag_count), [&](auto frag_range) {
                  for (size_t frag_idx = frag_range.begin(); frag_idx != frag_range.end();
                       ++frag_idx) {
                    auto& frag = fragments[frag_idx];

                    frag.offset =
                        frag_idx
                            ? ((frag_idx - 1) * table.fragment_size + first_frag_size)
                            : 0;
                    frag.row_count =
                        frag_idx
                            ? std::min(table.fragment_size,
                                       static_cast<size_t>(at->num_rows()) - frag.offset)
                            : first_frag_size;

                    auto meta = std::make_shared<ChunkMetadata>();
                    meta->type = col_info->type;
                    meta->numElements = frag.row_count;
                    if (col_type->isFixedLenArray()) {
                      meta->numBytes = frag.row_count * col_type->size();
                    } else if (col_type->isVarLenArray()) {
                      meta->numBytes =
                          computeTotalStringsLength(col_arr, frag.offset, frag.row_count);
                    } else {
                      meta->numBytes = frag.row_count * col_type->size();
                    }

                    computeStats(
                        col_arr->Slice(frag.offset, frag.row_count * elems_count),
                        col_type,
                        meta->chunkStats);
                    frag.metadata[col_idx] = meta;
                  }
                });  // each fragment
          } else {
            for (size_t frag_idx = 0; frag_idx < frag_count; ++frag_idx) {
              auto& frag = fragments[frag_idx];
              frag.offset =
                  frag_idx ? ((frag_idx - 1) * table.fragment_size + first_frag_size) : 0;
              frag.row_count =
                  frag_idx ? std::min(table.fragment_size,
                                      static_cast<size_t>(at->num_rows()) - frag.offset)
                           : first_frag_size;
              auto meta = std::make_shared<ChunkMetadata>();
              meta->type = col_info->type;
              meta->numElements = frag.row_count;
              CHECK(col_type->isText());
              meta->numBytes =
                  computeTotalStringsLength(col_arr, frag.offset, frag.row_count);
              meta->chunkStats.has_nulls =
                  col_arr->Slice(frag.offset, frag.row_count)->null_count();
              frag.metadata[col_idx] = meta;
            }
          }
        }
      });  // each column
  dict_lock.unlock();

  if (table.row_count) {
    // If table is not empty then we have to merge chunked arrays.
    CHECK_EQ(table.col_data.size(), col_data.size());
    for (size_t i = 0; i < table.col_data.size(); ++i) {
      arrow::ArrayVector lhs = table.col_data[i]->chunks();
      arrow::ArrayVector rhs = col_data[i]->chunks();
      lhs.insert(lhs.end(), rhs.begin(), rhs.end());
      table.col_data[i] = arrow::ChunkedArray::Make(std::move(lhs)).ValueOrDie();
    }

    // Probably need to merge the last existing fragment with the first new one.
    size_t start_frag = 0;
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      auto& first_frag = fragments.front();
      last_frag.row_count += first_frag.row_count;
      for (size_t col_idx = 0; col_idx < last_frag.metadata.size(); ++col_idx) {
        auto col_type = getColumnInfo(db_id_, table_id, col_idx + 1)->type;
        last_frag.metadata[col_idx]->numElements +=
            first_frag.metadata[col_idx]->numElements;
        last_frag.metadata[col_idx]->numBytes += first_frag.metadata[col_idx]->numBytes;
        mergeStats(last_frag.metadata[col_idx]->chunkStats,
                   first_frag.metadata[col_idx]->chunkStats,
                   col_type);
      }
      start_frag = 1;
    }

    // Copy the rest of fragments adjusting offset.
    table.fragments.reserve(table.fragments.size() + fragments.size() - start_frag);
    for (size_t frag_idx = start_frag; frag_idx < fragments.size(); ++frag_idx) {
      table.fragments.emplace_back(std::move(fragments[frag_idx]));
      table.fragments.back().offset += table.row_count;
    }

    table.row_count += at->num_rows();
  } else {
    CHECK_EQ(table.row_count, (size_t)0);
    table.col_data = std::move(col_data);
    table.fragments = std::move(fragments);
    table.row_count = at->num_rows();
  }

  getTableInfo(db_id_, table_id)->fragments = table.fragments.size();
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const std::vector<ColumnDescription>& columns,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  auto res = createTable(table_name, columns, options);
  appendCsvFile(file_name, table_name, parse_options);
  return res;
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  auto at = parseCsvFile(file_name, parse_options);
  return importArrowTable(at, table_name, options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 const std::string& table_name,
                                 const CsvParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendCsvFile(file_name, tinfo->table_id, parse_options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 int table_id,
                                 const CsvParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseCsvFile(file_name, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

void ArrowStorage::appendCsvData(const std::string& csv_data,
                                 const std::string& table_name,
                                 const CsvParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendCsvData(csv_data, tinfo->table_id, parse_options);
}

void ArrowStorage::appendCsvData(const std::string& csv_data,
                                 int table_id,
                                 const CsvParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseCsvData(csv_data, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

void ArrowStorage::appendJsonData(const std::string& json_data,
                                  const std::string& table_name,
                                  const JsonParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendJsonData(json_data, tinfo->table_id, parse_options);
}

void ArrowStorage::appendJsonData(const std::string& json_data,
                                  int table_id,
                                  const JsonParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseJsonData(json_data, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

TableInfoPtr ArrowStorage::importParquetFile(const std::string& file_name,
                                             const std::string& table_name,
                                             const TableOptions& options) {
  auto at = parseParquetFile(file_name);
  return importArrowTable(at, table_name, options);
}

void ArrowStorage::appendParquetFile(const std::string& file_name,
                                     const std::string& table_name) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendParquetFile(file_name, tinfo->table_id);
}

void ArrowStorage::appendParquetFile(const std::string& file_name, int table_id) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto at = parseParquetFile(file_name);
  appendArrowTable(at, table_id);
}

void ArrowStorage::dropTable(const std::string& table_name, bool throw_if_not_exist) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    if (throw_if_not_exist) {
      throw std::runtime_error("Cannot srop unknown table: "s + table_name);
    }
    return;
  }
  dropTable(tinfo->table_id);
}

void ArrowStorage::dropTable(int table_id, bool throw_if_not_exist) {
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);
  mapd_unique_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);

  if (!tables_.count(table_id)) {
    if (throw_if_not_exist) {
      throw std::runtime_error("Cannot drop table with invalid id: "s +
                               std::to_string(table_id));
    }
  }

  std::unique_ptr<TableData> table = std::move(tables_.at(table_id));
  mapd_unique_lock<mapd_shared_mutex> table_lock(table->mutex);
  tables_.erase(table_id);

  std::unordered_set<int> dicts_to_remove;
  auto col_infos = listColumnsNoLock(db_id_, table_id);
  for (auto& col_info : col_infos) {
    if (col_info->type->isExtDictionary()) {
      dicts_to_remove.insert(col_info->type->as<hdk::ir::ExtDictionaryType>()->dictId());
    }
  }

  SimpleSchemaProvider::dropTable(db_id_, table_id);
  // TODO: clean-up shared dictionaries without a full scan of
  // existing columns.
  for (auto& pr : column_infos_) {
    if (pr.second->type->isExtDictionary()) {
      dicts_to_remove.erase(pr.second->type->as<hdk::ir::ExtDictionaryType>()->dictId());
    }
  }

  for (auto dict_id : dicts_to_remove) {
    dicts_.erase(dict_id);
  }
}

void ArrowStorage::checkNewTableParams(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) const {
  if (columns.empty()) {
    throw std::runtime_error("Cannot create table with no columns");
  }

  if (table_name.empty()) {
    throw std::runtime_error("Cannot create table with empty name");
  }

  auto tinfo = getTableInfoNoLock(db_id_, table_name);
  if (tinfo) {
    throw std::runtime_error("Table with name '"s + table_name + "' already exists"s);
  }

  std::unordered_set<std::string> col_names;
  for (auto& col : columns) {
    if (col.name.empty()) {
      throw std::runtime_error("Empty column name is not allowed");
    }

    if (col.name == "rowid") {
      throw std::runtime_error("Reserved column name is not allowed: "s + col.name);
    }

    if (col_names.count(col.name)) {
      throw std::runtime_error("Duplicated column name: "s + col.name);
    }

    switch (col.type->id()) {
      case hdk::ir::Type::kBoolean:
      case hdk::ir::Type::kInteger:
      case hdk::ir::Type::kFloatingPoint:
      case hdk::ir::Type::kVarChar:
      case hdk::ir::Type::kText:
      case hdk::ir::Type::kDecimal:
      case hdk::ir::Type::kTime:
      case hdk::ir::Type::kDate:
      case hdk::ir::Type::kTimestamp:
      case hdk::ir::Type::kFixedLenArray:
      case hdk::ir::Type::kVarLenArray:
      case hdk::ir::Type::kInterval:
        break;
      case hdk::ir::Type::kExtDictionary: {
        auto dict_id = col.type->as<hdk::ir::ExtDictionaryType>()->dictId();
        if (dict_id > 0 && dicts_.count(dict_id) == 0) {
          throw std::runtime_error("Unknown dictionary ID is referenced in column '"s +
                                   col.name + "': "s + std::to_string(dict_id));
        }
      } break;
      default:
        throw std::runtime_error("Unsupported type for Arrow import: "s +
                                 col.type->toString());
    }

    col_names.insert(col.name);
  }
}

void ArrowStorage::compareSchemas(std::shared_ptr<arrow::Schema> lhs,
                                  std::shared_ptr<arrow::Schema> rhs) {
  auto& lhs_fields = lhs->fields();
  auto& rhs_fields = rhs->fields();
  if (lhs_fields.size() != rhs_fields.size()) {
    throw std::runtime_error("Mismatched clumns count");
  }

  for (size_t i = 0; i < lhs_fields.size(); ++i) {
    auto lhs_type = lhs_fields[i]->type();
    auto rhs_type = rhs_fields[i]->type();

    if (!lhs_type->Equals(rhs_type)) {
      throw std::runtime_error("Mismatched type for column "s + lhs_fields[i]->name() +
                               ": "s + lhs_type->ToString() + " vs. "s +
                               rhs_type->ToString());
    }
  }
}

void ArrowStorage::computeStats(std::shared_ptr<arrow::ChunkedArray> arr,
                                const hdk::ir::Type* type,
                                ChunkStats& stats) {
  auto elem_type =
      type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
  std::unique_ptr<Encoder> encoder(Encoder::Create(nullptr, elem_type));
  for (auto& chunk : arr->chunks()) {
    if (type->isVarLenArray()) {
      auto elem_size = elem_type->size();
      auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
      CHECK(chunk_list);
      auto offs = std::abs(chunk_list->value_offset(0)) / elem_size;
      auto len = std::abs(chunk_list->value_offset(chunk->length())) / elem_size - offs;
      auto elems = chunk_list->values();
      encoder->updateStatsEncoded(elems->data()->GetValues<int8_t>(
                                      1, (elems->data()->offset + offs) * type->size()),
                                  len);
    } else if (type->isFixedLenArray()) {
      encoder->updateStatsEncoded(
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * elem_type->size()),
          chunk->length(),
          true);
    } else {
      encoder->updateStatsEncoded(
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * elem_type->size()),
          chunk->length());
    }
  }

  encoder->fillChunkStats(stats, elem_type);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsvFile(
    const std::string& file_name,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) {
  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(file_name.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  return parseCsv(file_result.ValueOrDie(), parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsvData(
    const std::string& csv_data,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) {
  auto input = std::make_shared<arrow::io::BufferReader>(csv_data);
  return parseCsv(input, parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsv(
    std::shared_ptr<arrow::io::InputStream> input,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) {
  // by default arrow uses a process-wide memory pool, with jemalloc as the sys allocator
  // let's explore a per-table scheme with a fragment arena setup similar to
  // ArenaBufferMgr
  auto memory_pool = arena_memory_pool_.get();
  CHECK(memory_pool);
  // LOG(ERROR) << "Using " << memory_pool->backend_name() << " memory pool for csv
  // import.";
  auto io_context = arrow::io::IOContext(memory_pool);

  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.delimiter = parse_options.delimiter;

  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;
  arrow_read_options.block_size =
      parse_options.block_size;  // make this match fragment size?
  arrow_read_options.autogenerate_column_names =
      !parse_options.header && col_infos.empty();
  arrow_read_options.skip_rows = parse_options.header && !col_infos.empty()
                                     ? (parse_options.skip_rows + 1)
                                     : parse_options.skip_rows;

  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;
  arrow_convert_options.include_columns = arrow_read_options.column_names;
  arrow_convert_options.strings_can_be_null = true;

  for (auto& col_info : col_infos) {
    if (!col_info->is_rowid) {
      arrow_read_options.column_names.push_back(col_info->name);
      arrow_convert_options.column_types.emplace(
          col_info->name, getArrowImportType(ctx_, col_info->type));
    }
  }

  auto table_reader_result = arrow::csv::TableReader::Make(
      io_context, input, arrow_read_options, arrow_parse_options, arrow_convert_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> at;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    at = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow CSV in " << time << "ms";

  return at;
}

std::shared_ptr<arrow::Table> ArrowStorage::parseJsonData(
    const std::string& json_data,
    const JsonParseOptions parse_options,
    const ColumnInfoList& col_infos) {
  auto input = std::make_shared<arrow::io::BufferReader>(json_data);
  return parseJson(input, parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseJson(
    std::shared_ptr<arrow::io::InputStream> input,
    const JsonParseOptions parse_options,
    const ColumnInfoList& col_infos) {
  arrow::FieldVector fields;
  fields.reserve(col_infos.size());
  for (auto& col_info : col_infos) {
    if (!col_info->is_rowid) {
      fields.emplace_back(
          std::make_shared<arrow::Field>(col_info->name,
                                         getArrowImportType(ctx_, col_info->type),
                                         col_info->type->nullable()));
    }
  }
  auto schema = std::make_shared<arrow::Schema>(std::move(fields));

  auto arrow_parse_options = arrow::json::ParseOptions::Defaults();
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.explicit_schema = schema;

  auto arrow_read_options = arrow::json::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;
  arrow_read_options.block_size = parse_options.block_size;

  auto table_reader_result = arrow::json::TableReader::Make(
      arrow::default_memory_pool(), input, arrow_read_options, arrow_parse_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> at;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    at = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow JSON in " << time << "ms";

  return at;
}

std::shared_ptr<arrow::Table> ArrowStorage::parseParquetFile(
    const std::string& file_name) {
  auto file_result = arrow::io::ReadableFile::Open(file_name.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  auto inp = file_result.ValueOrDie();

  auto parquet_reader = parquet::ParquetFileReader::Open(inp);

  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  // Allow multithreding.
  parquet::ArrowReaderProperties prop(true);
  auto st = parquet::arrow::FileReader::Make(
      arrow::default_memory_pool(), std::move(parquet_reader), prop, &arrow_reader);
  if (!st.ok()) {
    throw std::runtime_error(st.ToString());
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  st = arrow_reader->ReadTable(&table);
  if (!st.ok()) {
    throw std::runtime_error(st.ToString());
  }

  return table;
}
