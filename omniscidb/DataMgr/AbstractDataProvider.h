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

#pragma once

#include "AbstractBufferMgr.h"

/**
 * This calss simply exists to hold all 'UNREACHABLE' definitions of
 * AbstractBufferMgr. This class should be removed when we have DataProvider
 * interface introduced.
 */
class AbstractDataProvider : public Data_Namespace::AbstractBufferMgr {
 public:
  AbstractDataProvider() : Data_Namespace::AbstractBufferMgr(0) {}

  Data_Namespace::AbstractBuffer* createBuffer(const ChunkKey& key,
                                               const size_t pageSize = 0,
                                               const size_t initialSize = 0) override {
    UNREACHABLE();
    return nullptr;
  }

  Data_Namespace::AbstractBuffer* createZeroCopyBuffer(
      const ChunkKey& key,
      std::unique_ptr<Data_Namespace::AbstractDataToken> token) override {
    UNREACHABLE();
    return nullptr;
  }

  std::unique_ptr<Data_Namespace::AbstractDataToken> getZeroCopyBufferMemory(
      const ChunkKey& key,
      size_t numBytes) override {
    return nullptr;
  }

  // TODO(dmitriim) remove this method after enabling
  // of hashtable, that takes into a count frag_id and offset
  std::unique_ptr<Data_Namespace::AbstractDataToken> getZeroCopyColumnData(
      const ColumnRef& col_ref) override {
    return nullptr;
  }

  void deleteBuffer(const ChunkKey& key, const bool purge = true) override {
    UNREACHABLE();
  }

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override {
    UNREACHABLE();
  }

  Data_Namespace::AbstractBuffer* getBuffer(const ChunkKey& key,
                                            const size_t numBytes = 0) override {
    UNREACHABLE();
    return nullptr;
  }

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override {
    UNREACHABLE();
  }

  bool isBufferOnDevice(const ChunkKey& key) override {
    UNREACHABLE();
    return false;
  }

  std::string printSlabs() override {
    UNREACHABLE();
    return "";
  }

  size_t getMaxSize() override {
    UNREACHABLE();
    return 0;
  }

  size_t getInUseSize() override {
    UNREACHABLE();
    return 0;
  }

  size_t getAllocated() override {
    UNREACHABLE();
    return 0;
  }

  bool isAllocationCapped() override {
    UNREACHABLE();
    return false;
  }

  const DictDescriptor* getDictMetadata(int dict_id, bool load_dict = true) override {
    UNREACHABLE();
    return nullptr;
  }

  Data_Namespace::AbstractBuffer* alloc(const size_t numBytes = 0) override {
    UNREACHABLE();
    return nullptr;
  }

  void free(Data_Namespace::AbstractBuffer* buffer) override { UNREACHABLE(); }

  MgrType getMgrType() override {
    UNREACHABLE();
    return static_cast<MgrType>(0);
  }

  std::string getStringMgrType() override {
    UNREACHABLE();
    return "";
  }

  size_t getNumChunks() override {
    UNREACHABLE();
    return 0;
  }
};
