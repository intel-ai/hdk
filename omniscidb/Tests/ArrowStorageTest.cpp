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

#include "ArrowStorage/ArrowStorage.h"

#include "TestHelpers.h"

#include <gtest/gtest.h>

#define EXPECT_THROW_WITH_MESSAGE(stmt, etype, whatstring) \
  EXPECT_THROW(                                            \
      try { stmt; } catch (const etype& ex) {              \
        EXPECT_EQ(std::string(ex.what()), whatstring);     \
        throw;                                             \
      },                                                   \
      etype)

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;

using namespace std::string_literals;

namespace {

int getDictId(const hdk::ir::Type* type) {
  if (type->isExtDictionary()) {
    return type->as<hdk::ir::ExtDictionaryType>()->dictId();
  }
  CHECK(type->isArray());
  return getDictId(type->as<hdk::ir::ArrayBaseType>()->elemType());
}

[[maybe_unused]] void dumpTableMeta(ArrowStorage& storage, int table_id) {
  std::cout << "Table #" << table_id << std::endl;

  std::cout << "  Schema:";
  auto col_infos = storage.listColumns(TEST_DB_ID, table_id);
  for (auto& col_info : col_infos) {
    std::cout << " " << col_info->name << "[" << col_info->column_id << "]("
              << col_info->type->toString() << ")";
  }
  std::cout << std::endl;

  std::cout << "  Fragments:" << std::endl;
  auto meta = storage.getTableMetadata(TEST_DB_ID, table_id);
  for (auto& frag : meta.fragments) {
    std::cout << "    Fragment #" << frag.fragmentId << " - " << frag.getNumTuples()
              << " row(s)" << std::endl;
    for (int col_id = 1; col_id < col_infos.back()->column_id; ++col_id) {
      auto& chunk_meta = frag.getChunkMetadataMap().at(col_id);
      std::cout << "      col" << col_id << " meta: " << chunk_meta->dump() << std::endl;
    }
  }
}

class TestBuffer : public Data_Namespace::AbstractBuffer {
 public:
  TestBuffer(size_t size) : Data_Namespace::AbstractBuffer(0) {
    size_ = size;
    data_.reset(new int8_t[size]);
  }

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int dst_device_id = -1) override {
    UNREACHABLE();
  }

  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int src_device_id = -1) override {
    UNREACHABLE();
  }

  void reserve(size_t num_bytes) override { CHECK_EQ(num_bytes, size_); }

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int device_id = -1) override {
    UNREACHABLE();
  }

  int8_t* getMemoryPtr() override { return data_.get(); }

  void setMemoryPtr(int8_t* new_ptr) override { UNREACHABLE(); }

  size_t pageCount() const override { return size_; }

  size_t pageSize() const override { return 1; }

  size_t reservedSize() const override { return size_; }

  MemoryLevel getType() const override { return MemoryLevel::CPU_LEVEL; }

 protected:
  std::unique_ptr<int8_t[]> data_;
};

std::string getFilePath(const std::string& file_name) {
  return TEST_SOURCE_PATH + "/ArrowStorageDataFiles/"s + file_name;
}

template <typename T>
std::vector<T> duplicate(const std::vector<T>& v) {
  std::vector<T> res;
  res.reserve(v.size() * 2);
  res.insert(res.end(), v.begin(), v.end());
  res.insert(res.end(), v.begin(), v.end());
  return res;
}

template <typename T>
std::vector<T> range(size_t size, T step) {
  std::vector<T> res(size);
  for (size_t i = 0; i < size; ++i) {
    res[i] = static_cast<T>((i + 1) * step);
  }
  return res;
}

void checkTableInfo(TableInfoPtr table_info,
                    int db_id,
                    int table_id,
                    const std::string& name,
                    size_t fragments) {
  CHECK_EQ(table_info->db_id, db_id);
  CHECK_EQ(table_info->table_id, table_id);
  CHECK_EQ(table_info->name, name);
  CHECK_EQ(table_info->fragments, fragments);
  CHECK_EQ(table_info->is_view, false);
  CHECK_EQ(table_info->persistence_level, Data_Namespace::MemoryLevel::CPU_LEVEL);
}

void checkColumnInfo(ColumnInfoPtr col_info,
                     int db_id,
                     int table_id,
                     int col_id,
                     const std::string& name,
                     const hdk::ir::Type* type,
                     bool is_rowid = false) {
  CHECK_EQ(col_info->db_id, db_id);
  CHECK_EQ(col_info->table_id, table_id);
  CHECK_EQ(col_info->column_id, col_id);
  CHECK_EQ(col_info->name, name);
  CHECK(col_info->type->equal(type));
  CHECK_EQ(col_info->is_rowid, is_rowid);
}

template <typename T>
void checkDatum(const Datum& actual, T expected, const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      ASSERT_EQ(static_cast<T>(actual.tinyintval), expected);
      break;
    case hdk::ir::Type::kExtDictionary:
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          ASSERT_EQ(static_cast<T>(actual.tinyintval), expected);
          break;
        case 2:
          ASSERT_EQ(static_cast<T>(actual.smallintval), expected);
          break;
        case 4:
          ASSERT_EQ(static_cast<T>(actual.intval), expected);
          break;
        case 8:
          ASSERT_EQ(actual.bigintval, static_cast<int64_t>(expected));
          break;
        default:
          CHECK(false);
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          CHECK_EQ(static_cast<T>(actual.floatval), expected);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          CHECK_EQ(static_cast<T>(actual.doubleval), expected);
          break;
        default:
          break;
      }
      break;
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
      ASSERT_EQ(actual.bigintval, static_cast<int64_t>(expected));
      break;
    default:
      break;
  }
}

void checkChunkMeta(std::shared_ptr<ChunkMetadata> meta,
                    const hdk::ir::Type* type,
                    size_t num_rows,
                    size_t num_bytes,
                    bool has_nulls) {
  CHECK(meta->type()->equal(type));
  CHECK_EQ(meta->numElements(), num_rows);
  CHECK_EQ(meta->numBytes(), num_bytes);
  CHECK_EQ(meta->chunkStats().has_nulls, has_nulls);
}

template <typename T>
void checkChunkMeta(std::shared_ptr<ChunkMetadata> meta,
                    const hdk::ir::Type* type,
                    size_t num_rows,
                    size_t num_bytes,
                    bool has_nulls,
                    T min,
                    T max) {
  checkChunkMeta(meta, type, num_rows, num_bytes, has_nulls);
  checkDatum(meta->chunkStats().min, min, type);
  checkDatum(meta->chunkStats().max, max, type);
}

template <typename T>
void checkFetchedData(ArrowStorage& storage,
                      int table_id,
                      int col_id,
                      int frag_id,
                      const std::vector<T>& expected,
                      const std::vector<int>& key_suffix = {}) {
  size_t buf_size = expected.size() * sizeof(T);
  TestBuffer dst(buf_size);
  ChunkKey key{TEST_DB_ID, table_id, col_id, frag_id};
  key.insert(key.end(), key_suffix.begin(), key_suffix.end());
  storage.fetchBuffer(key, &dst, buf_size);
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK_EQ(reinterpret_cast<const T*>(dst.getMemoryPtr())[i], expected[i]);
  }
}

template <typename T>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<T>& expected) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  std::vector<T> expected_chunk(expected.begin() + start_row, expected.begin() + end_row);
  auto cols = storage.listColumns(TEST_DB_ID, table_id);
  auto col_info = cols[col_idx];

  auto has_nulls = std::any_of(expected_chunk.begin(),
                               expected_chunk.end(),
                               [](const T& v) { return v == inline_null_value<T>(); });
  auto min =
      std::accumulate(expected_chunk.begin(),
                      expected_chunk.end(),
                      std::numeric_limits<T>::max(),
                      [](T min, T val) -> T {
                        return val == inline_null_value<T>() ? min : std::min(min, val);
                      });
  auto max =
      std::accumulate(expected_chunk.begin(),
                      expected_chunk.end(),
                      std::numeric_limits<T>::lowest(),
                      [](T max, T val) -> T {
                        return val == inline_null_value<T>() ? max : std::max(max, val);
                      });
  auto col_type = storage.getColumnInfo(TEST_DB_ID, table_id, col_info->column_id)->type;
  if (col_type->isDate()) {
    int64_t date_min = min == std::numeric_limits<T>::max()
                           ? std::numeric_limits<int64_t>::max()
                           : DateConverters::get_epoch_seconds_from_days(min);
    int64_t date_max = max == std::numeric_limits<T>::min()
                           ? std::numeric_limits<int64_t>::min()
                           : DateConverters::get_epoch_seconds_from_days(max);
    checkChunkMeta(chunk_meta_map.at(col_info->column_id),
                   col_type,
                   frag_rows,
                   frag_rows * sizeof(T),
                   has_nulls,
                   date_min,
                   date_max);
  } else {
    checkChunkMeta(chunk_meta_map.at(col_info->column_id),
                   col_type,
                   frag_rows,
                   frag_rows * sizeof(T),
                   has_nulls,
                   min,
                   max);
  }
  checkFetchedData(storage, table_id, col_info->column_id, frag_idx + 1, expected_chunk);
}

void checkStringColumnData(ArrowStorage& storage,
                           const ChunkMetadataMap& chunk_meta_map,
                           int table_id,
                           size_t row_count,
                           size_t fragment_size,
                           int col_id,
                           size_t frag_idx,
                           const std::vector<std::string>& vals) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  size_t chunk_size = 0;
  for (size_t i = start_row; i < end_row; ++i) {
    chunk_size += vals[i].size();
  }
  checkChunkMeta(chunk_meta_map.at(col_id),
                 storage.getColumnInfo(TEST_DB_ID, table_id, col_id)->type,
                 frag_rows,
                 chunk_size,
                 false);
  std::vector<int8_t> expected_data(chunk_size);
  std::vector<uint32_t> expected_offset(frag_rows + 1);
  uint32_t data_offset = 0;
  for (size_t i = start_row; i < end_row; ++i) {
    expected_offset[i - start_row] = data_offset;
    memcpy(expected_data.data() + data_offset, vals[i].data(), vals[i].size());
    data_offset += vals[i].size();
  }
  expected_offset.back() = data_offset;
  checkFetchedData(storage, table_id, col_id, frag_idx + 1, expected_offset, {2});
  checkFetchedData(storage, table_id, col_id, frag_idx + 1, expected_data, {1});
}

template <typename IndexType>
void checkStringDictColumnData(ArrowStorage& storage,
                               const ChunkMetadataMap& chunk_meta_map,
                               int table_id,
                               size_t row_count,
                               size_t fragment_size,
                               int col_id,
                               size_t frag_idx,
                               const std::vector<std::string>& expected) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;

  auto col_info = storage.getColumnInfo(TEST_DB_ID, table_id, col_id);
  auto& dict = *storage.getDictMetadata(getDictId(col_info->type))->stringDict;

  std::vector<IndexType> expected_ids(frag_rows);
  for (size_t i = start_row; i < end_row; ++i) {
    expected_ids[i - start_row] = static_cast<IndexType>(dict.getIdOfString(expected[i]));
  }

  checkChunkMeta(chunk_meta_map.at(col_id),
                 col_info->type,
                 frag_rows,
                 frag_rows * sizeof(IndexType),
                 false,
                 *std::min_element(expected_ids.begin(), expected_ids.end()),
                 *std::max_element(expected_ids.begin(), expected_ids.end()));

  checkFetchedData(storage, table_id, col_id, frag_idx + 1, expected_ids);
}

template <typename T>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::vector<T>>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto cols = storage.listColumns(TEST_DB_ID, table_id);
  auto col_info = cols[col_idx];

  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  size_t chunk_elems = 0;
  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  bool has_nulls = false;
  for (size_t i = start_row; i < end_row; ++i) {
    if (expected[i].empty() || !col_info->type->isVarLenArray() ||
        expected[i].front() != inline_null_array_value<T>()) {
      chunk_elems += expected[i].size();
    }
    for (auto& val : expected[i]) {
      if (val != inline_null_array_value<T>() && val != inline_null_value<T>()) {
        min = std::min(min, val);
        max = std::max(max, val);
      } else if (val == inline_null_value<T>()) {
        has_nulls = true;
      }
    }
  }
  size_t chunk_size = chunk_elems * sizeof(T);
  checkChunkMeta(chunk_meta_map.at(col_info->column_id),
                 col_info->type,
                 frag_rows,
                 chunk_size,
                 has_nulls,
                 min,
                 max);
  std::vector<T> expected_data(chunk_elems);
  std::vector<uint32_t> expected_offset(frag_rows + 1);
  uint32_t data_offset = 0;
  bool use_negative_offset = false;
  for (size_t i = start_row; i < end_row; ++i) {
    expected_offset[i - start_row] =
        use_negative_offset ? -data_offset * sizeof(T) : data_offset * sizeof(T);
    if (!expected[i].empty() && expected[i].front() == inline_null_array_value<T>() &&
        col_info->type->isVarLenArray()) {
      use_negative_offset = true;
    } else {
      use_negative_offset = false;
      memcpy(expected_data.data() + data_offset,
             expected[i].data(),
             expected[i].size() * sizeof(T));
      data_offset += expected[i].size();
    }
  }
  expected_offset.back() =
      use_negative_offset ? -data_offset * sizeof(T) : data_offset * sizeof(T);
  if (col_info->type->isVarLenArray()) {
    checkFetchedData(
        storage, table_id, col_info->column_id, frag_idx + 1, expected_offset, {2});
    checkFetchedData(
        storage, table_id, col_info->column_id, frag_idx + 1, expected_data, {1});
  } else {
    checkFetchedData(storage, table_id, col_info->column_id, frag_idx + 1, expected_data);
  }
}

template <>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::vector<std::string>>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto cols = storage.listColumns(TEST_DB_ID, table_id);
  auto col_info = cols[col_idx];
  auto& dict = *storage.getDictMetadata(getDictId(col_info->type))->stringDict;

  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  std::vector<int32_t> expected_ids;
  int32_t min = std::numeric_limits<int32_t>::min();
  int32_t max = std::numeric_limits<int32_t>::max();
  bool has_nulls = false;

  // varlen string arrays are not yet supported.
  CHECK(col_info->type->isFixedLenArray());
  size_t list_size = col_info->type->size() /
                     col_info->type->as<hdk::ir::ArrayBaseType>()->elemType()->size();
  size_t chunk_elems = frag_rows * list_size;
  for (size_t i = start_row; i < end_row; ++i) {
    if (expected[i].empty()) {
      expected_ids.push_back(inline_int_null_array_value<int32_t>());
      for (size_t j = 1; j < list_size; ++j) {
        expected_ids.push_back(inline_int_null_value<int32_t>());
      }
    } else {
      ASSERT_EQ(expected[i].size(), list_size);
      for (auto& val : expected[i]) {
        if (val == "<NULL>") {
          expected_ids.push_back(inline_int_null_value<int32_t>());
          has_nulls = true;
        } else {
          expected_ids.push_back(dict.getIdOfString(val));
          min = std::min(min, expected_ids.back());
          max = std::max(max, expected_ids.back());
        }
      }
    }
  }

  size_t chunk_size = chunk_elems * sizeof(int32_t);
  checkChunkMeta(chunk_meta_map.at(col_info->column_id),
                 col_info->type,
                 frag_rows,
                 chunk_size,
                 has_nulls,
                 min,
                 max);

  checkFetchedData(storage, table_id, col_info->column_id, frag_idx + 1, expected_ids);
}

template <>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::string>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto cols = storage.listColumns(TEST_DB_ID, table_id);
  auto col_info = cols[col_idx];
  if (col_info->type->isExtDictionary()) {
    switch (col_info->type->size()) {
      case 1:
        checkStringDictColumnData<int8_t>(storage,
                                          chunk_meta_map,
                                          table_id,
                                          row_count,
                                          fragment_size,
                                          col_info->column_id,
                                          frag_idx,
                                          expected);
        break;
      case 2:
        checkStringDictColumnData<int16_t>(storage,
                                           chunk_meta_map,
                                           table_id,
                                           row_count,
                                           fragment_size,
                                           col_info->column_id,
                                           frag_idx,
                                           expected);
        break;
      case 4:
        checkStringDictColumnData<int32_t>(storage,
                                           chunk_meta_map,
                                           table_id,
                                           row_count,
                                           fragment_size,
                                           col_info->column_id,
                                           frag_idx,
                                           expected);
        break;
      default:
        GTEST_FAIL();
    }
  } else {
    checkStringColumnData(storage,
                          chunk_meta_map,
                          table_id,
                          row_count,
                          fragment_size,
                          col_info->column_id,
                          frag_idx,
                          expected);
  }
}

void checkColumnData(ArrowStorage& storage,
                     const ChunkMetadataMap& chunk_meta_map,
                     int table_id,
                     size_t row_count,
                     size_t fragment_size,
                     size_t col_idx,
                     size_t frag_idx) {}

template <typename T, typename... Ts>
void checkColumnData(ArrowStorage& storage,
                     const ChunkMetadataMap& chunk_meta_map,
                     int table_id,
                     size_t row_count,
                     size_t fragment_size,
                     size_t col_idx,
                     size_t frag_idx,
                     const std::vector<T>& expected,
                     const std::vector<Ts>&... more_expected) {
  checkChunkData(storage,
                 chunk_meta_map,
                 table_id,
                 row_count,
                 fragment_size,
                 col_idx,
                 frag_idx,
                 expected);

  checkColumnData(storage,
                  chunk_meta_map,
                  table_id,
                  row_count,
                  fragment_size,
                  col_idx + 1,
                  frag_idx,
                  more_expected...);
}

template <typename... Ts>
void checkData(ArrowStorage& storage,
               int table_id,
               size_t row_count,
               size_t fragment_size,
               Ts... expected) {
  size_t frag_count = (row_count + fragment_size - 1) / fragment_size;
  auto meta = storage.getTableMetadata(TEST_DB_ID, table_id);
  auto cols = storage.listColumns(TEST_DB_ID, table_id);
  CHECK_EQ(meta.getNumTuples(), row_count);
  CHECK_EQ(meta.getPhysicalNumTuples(), row_count);
  CHECK_EQ(meta.fragments.size(), frag_count);
  for (size_t frag_idx = 0; frag_idx < frag_count; ++frag_idx) {
    size_t start_row = frag_idx * fragment_size;
    size_t end_row = std::min(row_count, start_row + fragment_size);
    size_t frag_rows = end_row - start_row;
    CHECK_EQ(meta.fragments[frag_idx].fragmentId, static_cast<int>(frag_idx + 1));
    CHECK_EQ(meta.fragments[frag_idx].physicalTableId, table_id);
    CHECK_EQ(meta.fragments[frag_idx].getNumTuples(), frag_rows);
    CHECK_EQ(meta.fragments[frag_idx].getPhysicalNumTuples(), frag_rows);

    auto chunk_meta_map = meta.fragments[frag_idx].getChunkMetadataMap();
    CHECK_EQ(chunk_meta_map.size(), sizeof...(Ts));
    for (int i = 0; i < static_cast<int>(chunk_meta_map.size()); ++i) {
      CHECK_EQ(chunk_meta_map.count(cols[i]->column_id), (size_t)1);
    }
    checkColumnData(storage,
                    chunk_meta_map,
                    table_id,
                    row_count,
                    fragment_size,
                    0,
                    frag_idx,
                    expected...);
  }
}

auto& ctx = hdk::ir::Context::defaultCtx();

}  // anonymous namespace

class ArrowStorageTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {}

  static void TearDownTestSuite() {}
};

TEST_F(ArrowStorageTest, CreateTable_OK) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo = storage.createTable(
      "table1", {{"col1", ctx.int32()}, {"col2", ctx.fp32()}, {"col3", ctx.fp64()}});
  checkTableInfo(tinfo, TEST_DB_ID, tinfo->table_id, "table1", 0);
  auto col_infos = storage.listColumns(*tinfo);
  CHECK_EQ(col_infos.size(), (size_t)4);
  checkColumnInfo(col_infos[0], TEST_DB_ID, tinfo->table_id, 1000, "col1", ctx.int32());
  checkColumnInfo(col_infos[1], TEST_DB_ID, tinfo->table_id, 1001, "col2", ctx.fp32());
  checkColumnInfo(col_infos[2], TEST_DB_ID, tinfo->table_id, 1002, "col3", ctx.fp64());
  checkColumnInfo(
      col_infos[3], TEST_DB_ID, tinfo->table_id, 1003, "rowid", ctx.int64(), true);
}

TEST_F(ArrowStorageTest, CreateTable_EmptyTableName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("", {{"col1", ctx.int32()}}), std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_DuplicatedTableName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_NO_THROW(storage.createTable("table1", {{"col1", ctx.int32()}}));
  ASSERT_THROW(storage.createTable("table1", {{"col1", ctx.int32()}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_NoColumns) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {}), std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_DuplicatedColumns) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(
      storage.createTable("table1", {{"col1", ctx.int32()}, {"col1", ctx.int32()}}),
      std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_EmptyColumnName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {{"", ctx.int32()}}), std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_ReservedColumnName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {{"rowid", ctx.int32()}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_SharedDict) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto type_dict0 = ctx.extDict(ctx.text(), 0);
  auto type_dict1 = ctx.extDict(ctx.text(), -1);
  auto type_dict2 = ctx.extDict(ctx.text(), -2);
  auto tinfo = storage.createTable("table1",
                                   {{"col1", type_dict0},
                                    {"col2", type_dict1},
                                    {"col3", type_dict2},
                                    {"col4", type_dict1},
                                    {"col5", type_dict2}});
  auto col_infos = storage.listColumns(*tinfo);
  CHECK_EQ(getDictId(col_infos[0]->type), addSchemaId(1, TEST_SCHEMA_ID));
  CHECK_EQ(getDictId(col_infos[1]->type), addSchemaId(2, TEST_SCHEMA_ID));
  CHECK_EQ(getDictId(col_infos[2]->type), addSchemaId(3, TEST_SCHEMA_ID));
  CHECK_EQ(getDictId(col_infos[3]->type), addSchemaId(2, TEST_SCHEMA_ID));
  CHECK_EQ(getDictId(col_infos[4]->type), addSchemaId(3, TEST_SCHEMA_ID));
  CHECK(storage.getDictMetadata(addSchemaId(1, TEST_SCHEMA_ID)));
  CHECK(storage.getDictMetadata(addSchemaId(2, TEST_SCHEMA_ID)));
  CHECK(storage.getDictMetadata(addSchemaId(2, TEST_SCHEMA_ID)));
}

TEST_F(ArrowStorageTest, CreateTable_WrongDictId) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto type = ctx.extDict(ctx.text(), 1);
  ASSERT_THROW(storage.createTable("table1", {{"col1", type}}), std::runtime_error);
}

TEST_F(ArrowStorageTest, DropTable) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo = storage.createTable("table1",
                                   {{"col1", ctx.int32()},
                                    {"col2", ctx.fp32()},
                                    {"col3", ctx.extDict(ctx.text(), 0)}});
  auto col_infos = storage.listColumns(*tinfo);
  storage.dropTable("table1");
  ASSERT_EQ(storage.getTableInfo(*tinfo), nullptr);
  for (auto& col_info : col_infos) {
    ASSERT_EQ(storage.getColumnInfo(*col_info), nullptr);
    if (col_info->type->isExtDictionary()) {
      ASSERT_EQ(storage.getDictMetadata(getDictId(col_info->type)), nullptr);
    }
  }
}

TEST_F(ArrowStorageTest, DropTable_SharedDicts) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo1 = storage.createTable("table1", {{"col1", ctx.extDict(ctx.text(), 0)}});
  auto col1_info = storage.getColumnInfo(*tinfo1, "col1");

  auto type_dict1 = ctx.extDict(ctx.text(), getDictId(col1_info->type));
  auto type_dict2 = ctx.extDict(ctx.text(), -1);
  auto tinfo2 = storage.createTable(
      "table2", {{"col1", type_dict1}, {"col2", type_dict2}, {"col3", type_dict2}});
  auto col2_info = storage.getColumnInfo(*tinfo2, "col2");

  auto col_infos = storage.listColumns(*tinfo2);
  storage.dropTable("table2");
  ASSERT_EQ(storage.getTableInfo(*tinfo2), nullptr);
  for (auto& col_info : col_infos) {
    ASSERT_EQ(storage.getColumnInfo(*col_info), nullptr);
  }

  ASSERT_NE(storage.getDictMetadata(getDictId(col1_info->type)), nullptr);
  ASSERT_EQ(storage.getDictMetadata(getDictId(col2_info->type)), nullptr);
}

void Test_ImportCsv_Numbers(const std::string& file_name,
                            const ArrowStorage::CsvParseOptions parse_options,
                            bool pass_schema,
                            size_t fragment_size = 32'000'000) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  if (pass_schema) {
    tinfo = storage.importCsvFile(getFilePath(file_name),
                                  "table1",
                                  {{"col1", ctx.int32()}, {"col2", ctx.fp32()}},
                                  table_options,
                                  parse_options);
    checkData(storage,
              tinfo->table_id,
              9,
              fragment_size,
              range(9, (int32_t)1),
              range(9, 10.0f));
  } else {
    tinfo = storage.importCsvFile(
        getFilePath(file_name), "table1", table_options, parse_options);
    checkData(
        storage, tinfo->table_id, 9, fragment_size, range(9, (int64_t)1), range(9, 10.0));
  }
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Header) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_NoHeader) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  Test_ImportCsv_Numbers("numbers_noheader.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Delim) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.delimiter = '|';
  Test_ImportCsv_Numbers("numbers_delim.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 5);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 2);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 20;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 20;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 5);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 2);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_Numbers_Header) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, false);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_Numbers_NoHeader) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  Test_ImportCsv_Numbers("numbers_noheader.csv", parse_options, false);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_NullsColumn_Header) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 32'000'000;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(
      getFilePath("nulls_header.csv"), "table1", table_options, parse_options);
  std::vector<double> col1(9, inline_null_value<double>());
  checkData(
      storage, tinfo->table_id, 9, table_options.fragment_size, col1, range(9, 10.0));
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_NullsColumn_Header) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 32'000'000;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(getFilePath("nulls_header.csv"),
                                             "table1",
                                             {{"col1", ctx.fp32()}, {"col2", ctx.fp32()}},
                                             table_options,
                                             parse_options);
  std::vector<float> col1(9, inline_null_value<float>());
  checkData(
      storage, tinfo->table_id, 9, table_options.fragment_size, col1, range(9, 10.0f));
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_NullsColumn_Append_Header) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;

  table_options.fragment_size = 32'000'000;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(
      getFilePath("nulls_header.csv"), "table1", table_options, parse_options);

  storage.appendCsvFile(getFilePath("nulls_header.csv"), "table1");
  std::vector<double> col1(18, inline_null_value<double>());
  auto inc = range(9, 10.0);
  inc.insert(inc.end(), inc.begin(), inc.end());
  checkData(storage, tinfo->table_id, 18, table_options.fragment_size, col1, inc);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_NullsColumn_ImportToCreated_Header) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);

  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", ctx.int32()}, {"col2", ctx.fp32()}});
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 32'000'000;
  ArrowStorage::CsvParseOptions parse_options;
  storage.appendCsvFile(getFilePath("nulls_header.csv"), "table1", parse_options);
  parse_options.header = false;
  storage.appendCsvData("80,100.0", tinfo->table_id, parse_options);
  std::vector<int32_t> col1(9, inline_null_value<int32_t>());
  col1.push_back(80);
  checkData(
      storage, tinfo->table_id, 10, table_options.fragment_size, col1, range(10, 10.0f));
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_NullsColumn_NullableTypeHeader) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);

  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"col1", ctx.int32(false)}, {"col2", ctx.fp32(false)}});
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 32'000'000;

  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData("1, 10.0", tinfo->table_id, parse_options);
  EXPECT_THROW_WITH_MESSAGE(
      storage.appendCsvFile(getFilePath("nulls_header.csv"), "table1"),
      std::runtime_error,
      "Null values used in non-nullable type: "s + ctx.int32(false)->toString());
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_NullsColumn_ImportCsvToCsv_Header) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);

  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 32'000'000;
  ArrowStorage::CsvParseOptions parse_options;

  TableInfoPtr tinfo = storage.importCsvFile(
      getFilePath("numbers_header.csv"), "table1", table_options, parse_options);

  storage.appendCsvFile(getFilePath("nulls_header.csv"), "table1", parse_options);
  std::vector<int64_t> col1 = range(9, (int64_t)1);
  std::vector<int64_t> colNull(9, inline_null_value<int64_t>());
  col1.insert(col1.end(), colNull.begin(), colNull.end());

  auto inc = range(9, 10.0);
  inc.insert(inc.end(), inc.begin(), inc.end());
  checkData(storage, tinfo->table_id, 18, table_options.fragment_size, col1, inc);
}

TEST_F(ArrowStorageTest, ImportCsv_DateTime) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(
      getFilePath("date_time.csv"), "table1", table_options, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_PartialSchema_Header1) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(getFilePath("numbers_header.csv"),
                                             "table1",
                                             {{"col1", ctx.int32()}},
                                             table_options,
                                             parse_options);
  checkData(storage,
            tinfo->table_id,
            9,
            table_options.fragment_size,
            range(9, (int32_t)1),
            range(9, 10.0));
}

TEST_F(ArrowStorageTest, ImportCsv_PartialSchema_Header2) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  TableInfoPtr tinfo = storage.importCsvFile(getFilePath("numbers_header.csv"),
                                             "table1",
                                             {{"col2", ctx.fp32()}},
                                             table_options,
                                             parse_options);
  checkData(storage,
            tinfo->table_id,
            9,
            table_options.fragment_size,
            range(9, (int64_t)1),
            range(9, 10.0f));
}

TEST_F(ArrowStorageTest, ImportCsv_PartialSchema_NoHeader1) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  TableInfoPtr tinfo = storage.importCsvFile(getFilePath("numbers_noheader.csv"),
                                             "table1",
                                             {{"col1", ctx.int32()}, {"col2", nullptr}},
                                             table_options,
                                             parse_options);
  checkData(storage,
            tinfo->table_id,
            9,
            table_options.fragment_size,
            range(9, (int32_t)1),
            range(9, 10.0));
}

TEST_F(ArrowStorageTest, ImportCsv_PartialSchema_NoHeader2) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  TableInfoPtr tinfo = storage.importCsvFile(getFilePath("numbers_noheader.csv"),
                                             "table1",
                                             {{"col1", nullptr}, {"col2", ctx.fp32()}},
                                             table_options,
                                             parse_options);
  checkData(storage,
            tinfo->table_id,
            9,
            table_options.fragment_size,
            range(9, (int64_t)1),
            range(9, 10.0f));
}

TEST_F(ArrowStorageTest, AppendCsvData) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", ctx.int32()}, {"col2", ctx.fp32()}});
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData("1,10.0\n2,20.0\n3,30.0\n", tinfo->table_id, parse_options);
  checkData(
      storage, tinfo->table_id, 3, 32'000'000, range(3, (int32_t)1), range(3, 10.0f));
}

void Test_AppendCsv_Numbers(size_t fragment_size) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  tinfo = storage.importCsvFile(getFilePath("numbers_header.csv"),
                                "table1",
                                {{"col1", ctx.int32()}, {"col2", ctx.fp32()}},
                                table_options,
                                parse_options);
  storage.appendCsvFile(getFilePath("numbers_header2.csv"), "table1");

  checkData(storage,
            tinfo->table_id,
            18,
            table_options.fragment_size,
            range(18, (int32_t)1),
            range(18, 10.0f));
}

TEST_F(ArrowStorageTest, AppendCsv_Numbers) {
  Test_AppendCsv_Numbers(100);
}

TEST_F(ArrowStorageTest, AppendCsv_Numbers_Multifrag) {
  Test_AppendCsv_Numbers(10);
  Test_AppendCsv_Numbers(5);
  Test_AppendCsv_Numbers(2);
  Test_AppendCsv_Numbers(1);
}

void Test_ImportCsv_Strings(bool pass_schema,
                            bool read_twice,
                            const ArrowStorage::CsvParseOptions& parse_options,
                            size_t fragment_size = 32'000'000) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  if (pass_schema) {
    tinfo = storage.importCsvFile(getFilePath("strings.csv"),
                                  "table1",
                                  {{"col1", ctx.text()}, {"col2", ctx.text()}},
                                  table_options,
                                  parse_options);
  } else {
    tinfo = storage.importCsvFile(
        getFilePath("strings.csv"), "table1", table_options, parse_options);
  }

  if (read_twice) {
    storage.appendCsvFile(getFilePath("strings.csv"), "table1", parse_options);
  }

  std::vector<std::string> col1_expected = {"s1"s, "ss2"s, "sss3"s, "ssss4"s, "sssss5"s};
  std::vector<std::string> col2_expected = {
      "dd1"s, "dddd2"s, "dddddd3"s, "dddddddd4"s, "dddddddddd5"s};
  if (read_twice) {
    col1_expected = duplicate(col1_expected);
    col2_expected = duplicate(col2_expected);
  }
  checkData(storage,
            tinfo->table_id,
            read_twice ? 10 : 5,
            table_options.fragment_size,
            col1_expected,
            col2_expected);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, false, parse_options, 3);
  Test_ImportCsv_Strings(true, false, parse_options, 2);
  Test_ImportCsv_Strings(true, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, false, parse_options, 3);
  Test_ImportCsv_Strings(true, false, parse_options, 2);
  Test_ImportCsv_Strings(true, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_NoSchema) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(false, false, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, true, parse_options, 7);
  Test_ImportCsv_Strings(true, true, parse_options, 5);
  Test_ImportCsv_Strings(true, true, parse_options, 3);
  Test_ImportCsv_Strings(true, true, parse_options, 2);
  Test_ImportCsv_Strings(true, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, true, parse_options, 7);
  Test_ImportCsv_Strings(true, true, parse_options, 5);
  Test_ImportCsv_Strings(true, true, parse_options, 3);
  Test_ImportCsv_Strings(true, true, parse_options, 2);
  Test_ImportCsv_Strings(true, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_NoSchema) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(false, true, parse_options);
}

void Test_ImportCsv_Dict(bool shared_dict,
                         bool read_twice,
                         const ArrowStorage::CsvParseOptions& parse_options,
                         size_t fragment_size = 32'000'000,
                         int index_size = 4) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  auto dict1_type = ctx.extDict(ctx.text(), shared_dict ? -1 : 0, index_size);
  tinfo = storage.importCsvFile(getFilePath("strings.csv"),
                                "table1",
                                {{"col1", dict1_type}, {"col2", dict1_type}},
                                table_options,
                                parse_options);
  if (read_twice) {
    storage.appendCsvFile(getFilePath("strings.csv"), "table1", parse_options);
  }

  if (shared_dict) {
    auto col1_info = storage.getColumnInfo(*tinfo, "col1");
    auto dict1 =
        storage
            .getDictMetadata(col1_info->type->as<hdk::ir::ExtDictionaryType>()->dictId())
            ->stringDict;
    CHECK_EQ(dict1->storageEntryCount(), (size_t)10);
  } else {
    auto col1_info = storage.getColumnInfo(*tinfo, "col1");
    auto dict1 =
        storage
            .getDictMetadata(col1_info->type->as<hdk::ir::ExtDictionaryType>()->dictId())
            ->stringDict;
    CHECK_EQ(dict1->storageEntryCount(), (size_t)5);
    auto col2_info = storage.getColumnInfo(*tinfo, "col2");
    auto dict2 =
        storage
            .getDictMetadata(col2_info->type->as<hdk::ir::ExtDictionaryType>()->dictId())
            ->stringDict;
    CHECK_EQ(dict2->storageEntryCount(), (size_t)5);
  }

  std::vector<std::string> col1_expected = {"s1"s, "ss2"s, "sss3"s, "ssss4"s, "sssss5"s};
  std::vector<std::string> col2_expected = {
      "dd1"s, "dddd2"s, "dddddd3"s, "dddddddd4"s, "dddddddddd5"s};
  if (read_twice) {
    col1_expected = duplicate(col1_expected);
    col2_expected = duplicate(col2_expected);
  }
  checkData(storage,
            tinfo->table_id,
            read_twice ? 10 : 5,
            table_options.fragment_size,
            col1_expected,
            col2_expected);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options, 5);
  Test_ImportCsv_Dict(false, false, parse_options, 3);
  Test_ImportCsv_Dict(false, false, parse_options, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, false, parse_options, 5);
  Test_ImportCsv_Dict(false, false, parse_options, 3);
  Test_ImportCsv_Dict(false, false, parse_options, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_SharedDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_SmallDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options, 100, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 3, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 2, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_SharedSmallDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, true, parse_options, 100, 2);
  Test_ImportCsv_Dict(true, true, parse_options, 3, 2);
  Test_ImportCsv_Dict(true, true, parse_options, 2, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, true, parse_options, 5);
  Test_ImportCsv_Dict(false, true, parse_options, 3);
  Test_ImportCsv_Dict(false, true, parse_options, 2);
  Test_ImportCsv_Dict(false, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, true, parse_options, 5);
  Test_ImportCsv_Dict(false, true, parse_options, 3);
  Test_ImportCsv_Dict(false, true, parse_options, 2);
  Test_ImportCsv_Dict(false, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_SharedDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendJsonData) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"col1", ctx.int32()}, {"col2", ctx.fp32()}, {"col3", ctx.text()}});
  storage.appendJsonData(R"___({"col1": 1, "col2": 10.0, "col3": "s1"}
{"col1": 2, "col2": 20.0, "col3": "s2"}
{"col1": 3, "col2": 30.0, "col3": "s3"})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            range(3, (int32_t)1),
            range(3, 10.0f),
            std::vector<std::string>({"s1"s, "s2"s, "s3"s}));
}

TEST_F(ArrowStorageTest, AppendJsonData_Arrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto int_array = ctx.arrayVarLen(ctx.int32());
  auto float_array = ctx.arrayVarLen(ctx.fp32());
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int_array}, {"col2", float_array}});
  storage.appendJsonData(R"___({"col1": [1, 2, 3], "col2": [10.0, 20.0, 30.0]}
{"col1": [4, 5, 6], "col2": [40.0, 50.0, 60.0]}
{"col1": [7, 8, 9], "col2": [70.0, 80.0, 90.0]})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<std::vector<int>>({std::vector<int>({1, 2, 3}),
                                           std::vector<int>({4, 5, 6}),
                                           std::vector<int>({7, 8, 9})}),
            std::vector<std::vector<float>>({std::vector<float>({10.0f, 20.0f, 30.0f}),
                                             std::vector<float>({40.0f, 50.0f, 60.0f}),
                                             std::vector<float>({70.0f, 80.0f, 90.0f})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_ArraysWithNulls) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto int_array = ctx.arrayVarLen(ctx.int32());
  auto float_array = ctx.arrayVarLen(ctx.fp32());
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int_array}, {"col2", float_array}});
  storage.appendJsonData(R"___({"col1": [1, 2, 3], "col2": [null, 20.0, 30.0]}
{"col1": null, "col2": [40.0, null, 60.0]}
{"col1": [7, 8, 9], "col2": null})___",
                         tinfo->table_id);
  checkData(
      storage,
      tinfo->table_id,
      3,
      32'000'000,
      std::vector<std::vector<int>>({std::vector<int>({1, 2, 3}),
                                     std::vector<int>({inline_null_array_value<int>()}),
                                     std::vector<int>({7, 8, 9})}),
      std::vector<std::vector<float>>(
          {std::vector<float>({inline_null_value<float>(), 20.0f, 30.0f}),
           std::vector<float>({40.0f, inline_null_value<float>(), 60.0f}),
           std::vector<float>({inline_null_array_value<float>()})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_FixedSizeArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto int3_array = ctx.arrayFixed(3, ctx.int32());
  auto float2_array = ctx.arrayFixed(2, ctx.fp32());
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int3_array}, {"col2", float2_array}});
  storage.appendJsonData(R"___({"col1": [1, 2], "col2": [null, 20.0]}
{"col1": null, "col2": [40.0, null, 60.0, 70.0]}
{"col1": [7, 8, 9, 10], "col2": null}
{"col1": [11, 12, 13], "col2": [110.0]})___",
                         tinfo->table_id);
  checkData(
      storage,
      tinfo->table_id,
      4,
      32'000'000,
      std::vector<std::vector<int>>({std::vector<int>({1, 2, inline_null_value<int>()}),
                                     std::vector<int>({inline_null_array_value<int>(),
                                                       inline_null_value<int>(),
                                                       inline_null_value<int>()}),
                                     std::vector<int>({7, 8, 9}),
                                     std::vector<int>({11, 12, 13})}),
      std::vector<std::vector<float>>(
          {std::vector<float>({inline_null_value<float>(), 20.0f}),
           std::vector<float>({40.0f, inline_null_value<float>()}),
           std::vector<float>(
               {inline_null_array_value<float>(), inline_null_value<float>()}),
           std::vector<float>({110.f, inline_null_value<float>()})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_StringFixedSizeArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto string3_array_type = ctx.arrayFixed(3, ctx.extDict(ctx.text(), 0));
  TableInfoPtr tinfo = storage.createTable("table1", {{"col1", string3_array_type}});
  storage.appendJsonData(R"___({"col1": ["str1", "str2"]}
{"col1": null}
{"col1": ["str2", "str8", "str3", "str10"]}
{"col1": ["str1", null, "str3"]})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            4,
            32'000'000,
            std::vector<std::vector<std::string>>(
                {std::vector<std::string>({"str1"s, "str2"s, "<NULL>"}),
                 std::vector<std::string>({}),
                 std::vector<std::string>({"str2"s, "str8"s, "str3"s}),
                 std::vector<std::string>({"str1"s, "<NULL>", "str3"s})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_DateTime) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo =
      storage.createTable("table1",
                          {{"ts_0", ctx.timestamp(hdk::ir::TimeUnit::kSecond)},
                           {"ts_3", ctx.timestamp(hdk::ir::TimeUnit::kMilli)},
                           {"ts_6", ctx.timestamp(hdk::ir::TimeUnit::kMicro)},
                           {"ts_9", ctx.timestamp(hdk::ir::TimeUnit::kNano)},
                           {"t", ctx.time64(hdk::ir::TimeUnit::kSecond)},
                           {"d", ctx.date32(hdk::ir::TimeUnit::kDay)},
                           {"o1", ctx.date16(hdk::ir::TimeUnit::kDay)},
                           {"o2", ctx.date32(hdk::ir::TimeUnit::kDay)}});
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData(
      "2014-12-13 22:23:15,2014-12-13 22:23:15.323,1999-07-11 14:02:53.874533,2006-04-26 "
      "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09",
      tinfo->table_id,
      parse_options);
  storage.appendCsvData(
      "2014-12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 22:23:15.874533,2014-12-13 "
      "22:23:15.607435763,15:13:14,,,",
      tinfo->table_id,
      parse_options);
  storage.appendCsvData(
      "2014-12-14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 22:23:15.437321,2014-12-14 "
      "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09",
      tinfo->table_id,
      parse_options);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<int64_t>({1418509395, 1418509395, 1418595795}),
            std::vector<int64_t>({1418509395323, 1418509395323, 1418595795750}),
            std::vector<int64_t>({931701773874533, 1418509395874533, 1418595795437321}),
            std::vector<int64_t>(
                {1146023344607435125, 1418509395607435763, 1418595795934567401}),
            std::vector<int64_t>({54794, 54794, 54794}),
            std::vector<int32_t>({10843, inline_null_value<int32_t>(), 10843}),
            std::vector<int16_t>({10843, inline_null_value<int16_t>(), 10843}),
            std::vector<int32_t>({10843, inline_null_value<int32_t>(), 10843}));
}

TEST_F(ArrowStorageTest, AppendJsonData_DateTime_Multifrag) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo =
      storage.createTable("table1",
                          {{"ts_0", ctx.timestamp(hdk::ir::TimeUnit::kSecond)},
                           {"ts_3", ctx.timestamp(hdk::ir::TimeUnit::kMilli)},
                           {"ts_6", ctx.timestamp(hdk::ir::TimeUnit::kMicro)},
                           {"ts_9", ctx.timestamp(hdk::ir::TimeUnit::kNano)},
                           {"t", ctx.time64(hdk::ir::TimeUnit::kSecond)},
                           {"d", ctx.date32(hdk::ir::TimeUnit::kDay)},
                           {"o1", ctx.date16(hdk::ir::TimeUnit::kDay)},
                           {"o2", ctx.date32(hdk::ir::TimeUnit::kDay)}},
                          table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  for (int i = 0; i < 4; ++i) {
    storage.appendCsvData(
        "2014-12-13 22:23:15,2014-12-13 22:23:15.323,1999-07-11 "
        "14:02:53.874533,2006-04-26 "
        "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09",
        tinfo->table_id,
        parse_options);
  }
  for (int i = 0; i < 3; ++i) {
    storage.appendCsvData(
        "2014-12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 "
        "22:23:15.874533,2014-12-13 "
        "22:23:15.607435763,15:13:14,,,",
        tinfo->table_id,
        parse_options);
  }
  for (int i = 0; i < 3; ++i) {
    storage.appendCsvData(
        "2014-12-14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 "
        "22:23:15.437321,2014-12-14 "
        "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09",
        tinfo->table_id,
        parse_options);
  }
  checkData(storage,
            tinfo->table_id,
            10,
            2,
            std::vector<int64_t>({1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418595795,
                                  1418595795,
                                  1418595795}),
            std::vector<int64_t>({1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418595795750,
                                  1418595795750,
                                  1418595795750}),
            std::vector<int64_t>({931701773874533,
                                  931701773874533,
                                  931701773874533,
                                  931701773874533,
                                  1418509395874533,
                                  1418509395874533,
                                  1418509395874533,
                                  1418595795437321,
                                  1418595795437321,
                                  1418595795437321}),
            std::vector<int64_t>({1146023344607435125,
                                  1146023344607435125,
                                  1146023344607435125,
                                  1146023344607435125,
                                  1418509395607435763,
                                  1418509395607435763,
                                  1418509395607435763,
                                  1418595795934567401,
                                  1418595795934567401,
                                  1418595795934567401}),
            std::vector<int64_t>(
                {54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794}),
            std::vector<int32_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  10843,
                                  10843,
                                  10843}),
            std::vector<int16_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int16_t>(),
                                  inline_null_value<int16_t>(),
                                  inline_null_value<int16_t>(),
                                  10843,
                                  10843,
                                  10843}),
            std::vector<int32_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  10843,
                                  10843,
                                  10843}));
}

TEST_F(ArrowStorageTest, AppendCsvData_BoolWithNulls) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"b1", ctx.boolean()}, {"b2", ctx.boolean()}}, table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData("true,true", tinfo->table_id, parse_options);
  storage.appendCsvData("true,false", tinfo->table_id, parse_options);
  storage.appendCsvData("false,true", tinfo->table_id, parse_options);
  storage.appendCsvData(",true", tinfo->table_id, parse_options);
  storage.appendCsvData(",false", tinfo->table_id, parse_options);
  storage.appendCsvData("true,", tinfo->table_id, parse_options);
  storage.appendCsvData("false,", tinfo->table_id, parse_options);
  storage.appendCsvData(",", tinfo->table_id, parse_options);
  checkData(storage,
            tinfo->table_id,
            8,
            2,
            std::vector<int8_t>({1, 1, 0, -128, -128, 1, 0, -128}),
            std::vector<int8_t>({1, 0, 1, 1, 0, -128, -128, -128}));
}

TEST_F(ArrowStorageTest, AppendCsvData_BoolWithNulls_SingleChunk) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"b1", ctx.boolean()}, {"b2", ctx.boolean()}}, table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData(
      "true,true\n"
      "true,false\n"
      "false,true\n"
      ",true\n"
      ",false\n"
      "true,\n"
      "false,\n"
      ",",
      tinfo->table_id,
      parse_options);
  checkData(storage,
            tinfo->table_id,
            8,
            2,
            std::vector<int8_t>({1, 1, 0, -128, -128, 1, 0, -128}),
            std::vector<int8_t>({1, 0, 1, 1, 0, -128, -128, -128}));
}

TEST_F(ArrowStorageTest, AppendJsonData_BoolArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto bool_3 = ctx.arrayFixed(3, ctx.boolean());
  auto bool_any = ctx.arrayVarLen(ctx.boolean());
  TableInfoPtr tinfo = storage.createTable("table1", {{"b1", bool_3}, {"b2", bool_any}});
  storage.appendJsonData("{\"b1\": [true, true, true], \"b2\": [true, false]}",
                         tinfo->table_id);
  storage.appendJsonData("{\"b1\": [false, false, false], \"b2\": [false, true]}",
                         tinfo->table_id);
  storage.appendJsonData("{\"b1\": [null, false, true], \"b2\": [null]}",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<std::vector<int8_t>>({std::vector<int8_t>({1, 1, 1}),
                                              std::vector<int8_t>({0, 0, 0}),
                                              std::vector<int8_t>({-128, 0, 1})}),
            std::vector<std::vector<int8_t>>({std::vector<int8_t>({1, 0}),
                                              std::vector<int8_t>({0, 1}),
                                              std::vector<int8_t>({-128})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_IntArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto smallint_any = ctx.arrayVarLen(ctx.int16());
  TableInfoPtr tinfo = storage.createTable("table1", {{"siv", smallint_any}});
  storage.appendJsonData("{\"siv\": [1, 2, 3]}", tinfo->table_id);
  storage.appendJsonData("{\"siv\": [2]}", tinfo->table_id);
  storage.appendJsonData("{\"siv\": [2, 4]}", tinfo->table_id);
  storage.appendJsonData("{\"siv\": [4, 5, 6]}", tinfo->table_id);
  storage.appendJsonData("{\"siv\": [5]}", tinfo->table_id);
  storage.appendJsonData("{\"siv\": [5, 7]}", tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            6,
            32'000'000,
            std::vector<std::vector<int16_t>>({std::vector<int16_t>({1, 2, 3}),
                                               std::vector<int16_t>({2}),
                                               std::vector<int16_t>({2, 4}),
                                               std::vector<int16_t>({4, 5, 6}),
                                               std::vector<int16_t>({5}),
                                               std::vector<int16_t>({5, 7})}));
}

TEST_F(ArrowStorageTest, AppendCsvData_Decimals) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"d1", ctx.decimal64(10, 2)}, {"d2", ctx.decimal64(10, 4)}});
  storage.appendCsvData("1.1,2.22", tinfo->table_id, parse_options);
  storage.appendCsvData("1.11,2.2222", tinfo->table_id, parse_options);
  storage.appendCsvData("1,2", tinfo->table_id, parse_options);
  storage.appendCsvData(",", tinfo->table_id, parse_options);
  checkData(storage,
            tinfo->table_id,
            4,
            32'000'000,
            std::vector<int64_t>({110, 111, 100, inline_null_value<int64_t>()}),
            std::vector<int64_t>({22200, 22222, 20000, inline_null_value<int64_t>()}));
}

TEST_F(ArrowStorageTest, AppendJsonData_DecimalArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto decimal_3 = ctx.arrayFixed(3, ctx.decimal64(10, 2));
  auto decimal_any = ctx.arrayVarLen(ctx.decimal64(10, 4));
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"d1", decimal_3}, {"d2", decimal_any}});
  storage.appendJsonData(R"___({"d1": [1.1, 2.2, 3.3], "d2": [1.11, 2.22, 3.33]}
{"d1": [10.10, null, 30.30], "d2": null}
{"d1": null, "d2": [10.1010, 20.2020, null]})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<std::vector<int64_t>>(
                {std::vector<int64_t>({110, 220, 330}),
                 std::vector<int64_t>({1010, inline_null_value<int64_t>(), 3030}),
                 std::vector<int64_t>({inline_null_array_value<int64_t>(),
                                       inline_null_value<int64_t>(),
                                       inline_null_value<int64_t>()})}),
            std::vector<std::vector<int64_t>>(
                {std::vector<int64_t>({11100, 22200, 33300}),
                 std::vector<int64_t>({inline_null_array_value<int64_t>()}),
                 std::vector<int64_t>({101010, 202020, inline_null_value<int64_t>()})}));
}

TEST_F(ArrowStorageTest, ImportParquet) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo = storage.importParquetFile(getFilePath("int_float.parquet"), "table1");

  checkData(storage,
            tinfo->table_id,
            5,
            32'000'000,
            std::vector<int64_t>({1, 2, 3, 4, 5}),
            std::vector<double>({1.1, 2.2, 3.3, 4.4, 5.5}));
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
