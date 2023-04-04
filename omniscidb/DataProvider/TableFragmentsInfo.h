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

#pragma once

#include "DataMgr/ChunkMetadata.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"

#include <deque>
#include <list>
#include <map>
#include <mutex>

class ResultSet;

/**
 * @class FragmentInfo
 * @brief Used by Fragmenter classes to store info about each
 * fragment - the fragment id and number of tuples(rows)
 * currently stored by that fragment
 */

class FragmentInfo {
 public:
  FragmentInfo()
      : fragmentId(-1)
      , shadowNumTuples(0)
      , physicalTableId(-1)
      , resultSet(nullptr)
      , numTuples(0)
      , synthesizedNumTuplesIsValid(false) {}

  void setChunkMetadataMap(const ChunkMetadataMap& chunk_metadata_map) {
    this->chunkMetadataMap = chunk_metadata_map;
  }

  void setChunkMetadata(const int col, std::shared_ptr<ChunkMetadata> chunkMetadata) {
    chunkMetadataMap[col] = chunkMetadata;
  }

  const ChunkMetadataMap& getChunkMetadataMap() const;

  const ChunkMetadataMap& getChunkMetadataMapPhysical() const { return chunkMetadataMap; }

  ChunkMetadataMap getChunkMetadataMapPhysicalCopy() const;

  size_t getNumTuples() const;

  size_t getPhysicalNumTuples() const { return numTuples; }

  bool isEmptyPhysicalFragment() const { return physicalTableId >= 0 && !numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  void invalidateNumTuples() const { synthesizedNumTuplesIsValid = false; }

  int fragmentId;
  size_t shadowNumTuples;
  std::vector<int> deviceIds;
  int physicalTableId;
  ChunkMetadataMap shadowChunkMetadataMap;
  mutable ResultSet* resultSet;
  mutable std::shared_ptr<std::mutex> resultSetMutex;

 private:
  mutable size_t numTuples;
  mutable ChunkMetadataMap chunkMetadataMap;
  mutable bool synthesizedNumTuplesIsValid;
};

class TableFragmentsInfo {
 public:
  TableFragmentsInfo() : numTuples(0) {}

  size_t getNumTuples() const;

  size_t getNumTuplesUpperBound() const;

  size_t getPhysicalNumTuples() const { return numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  size_t getFragmentNumTuplesUpperBound() const;

  std::vector<int> chunkKeyPrefix;
  std::vector<FragmentInfo> fragments;

 private:
  mutable size_t numTuples;
};
