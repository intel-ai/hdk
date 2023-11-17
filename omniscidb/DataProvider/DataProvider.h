/*
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

#pragma once

#include <memory>

#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/MemoryLevel.h"
#include "SchemaMgr/ColumnInfo.h"
#include "Shared/types.h"

#include "DictDescriptor.h"
#include "TableFragmentsInfo.h"

class DataProvider {
 public:
  // fetch data
  virtual std::shared_ptr<Chunk_NS::Chunk> getChunk(
      ColumnInfoPtr col_info,
      const ChunkKey& key,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      const size_t num_bytes,
      const size_t num_elems) = 0;

  // CPU only
  // TODO(dmitriim) remove this method after enabling of hashtable
  virtual std::unique_ptr<Data_Namespace::AbstractDataToken> getZeroCopyColumnData(
      const ColumnRef& col_ref) = 0;

  virtual std::shared_ptr<const TableFragmentsInfo> getTableMetadata(
      int db_id,
      int table_id) const = 0;

  virtual const DictDescriptor* getDictMetadata(int dict_id,
                                                bool load_dict = true) const = 0;

  virtual ~DataProvider() = default;
};

using DataProviderPtr = std::shared_ptr<DataProvider>;
