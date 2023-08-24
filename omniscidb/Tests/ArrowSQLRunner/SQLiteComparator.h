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

#pragma once

#include "ResultSet/ArrowResultSet.h"
#include "ResultSetRegistry/ResultSetTableToken.h"
#include "Shared/sqltypes.h"
#include "SqliteConnector/SqliteConnector.h"

namespace TestHelpers {

constexpr double EPS = 1.25e-5;

class SQLiteComparator {
 public:
  SQLiteComparator(bool use_row_iterator = true);
  ~SQLiteComparator();

  void query(const std::string& query_string) { connector_.query(query_string); }

  void batch_insert(const std::string& table_name,
                    std::vector<std::vector<std::string>>& insert_vals) {
    connector_.batch_insert(table_name, insert_vals);
  }

  void compare(hdk::ResultSetTableTokenPtr omnisci_results,
               const std::string& query_string,
               const ExecutorDeviceType device_type);

  void compare_arrow_output(std::unique_ptr<ArrowResultSet>& arrow_omnisci_results,
                            const std::string& sqlite_query_string,
                            const ExecutorDeviceType device_type);

  // added to deal with time shift for now testing
  void compare_timstamp_approx(hdk::ResultSetTableTokenPtr omnisci_results,
                               const std::string& query_string,
                               const ExecutorDeviceType device_type);

 private:
  std::string sqlite_db_dir_;
  SqliteConnector connector_;
  bool use_row_iterator_;
};

}  // namespace TestHelpers
