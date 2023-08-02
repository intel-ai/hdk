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

#ifndef QUERYENGINE_TABLEGENERATIONS_H
#define QUERYENGINE_TABLEGENERATIONS_H

#include <cstdint>
#include <unordered_map>

struct TableGeneration {
  int64_t tuple_count;
  int64_t start_rowid;
};

class TableGenerations {
 public:
  void setGeneration(int db_id, int table_id, const TableGeneration& generation);

  const TableGeneration& getGeneration(int db_id, int table_id) const;

  const std::unordered_map<std::pair<int, int>, TableGeneration>& asMap() const;

  void clear();

 private:
  std::unordered_map<std::pair<int, int>, TableGeneration> id_to_generation_;
};

#endif  // QUERYENGINE_TABLEGENERATIONS_H
