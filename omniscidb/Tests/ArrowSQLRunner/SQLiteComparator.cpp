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

#include "SQLiteComparator.h"
#include "IR/Context.h"
#include "IR/Type.h"

#include "Shared/DateTimeParser.h"

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include <boost/algorithm/string.hpp>
#include <filesystem>

#ifdef _WIN32
#define timegm _mkgmtime
#endif

using namespace std::string_literals;

namespace TestHelpers {

namespace {

void checkTypeConsistency(const int ref_col_type, const hdk::ir::Type* col_type) {
  if (ref_col_type == SQLITE_NULL) {
    // TODO(alex): re-enable the check that col_type is nullable,
    //             got invalidated because of outer joins
    return;
  }
  if (col_type->isInteger()) {
    CHECK_EQ(SQLITE_INTEGER, ref_col_type);
  } else if (col_type->isFloatingPoint() || col_type->isDecimal()) {
    CHECK(ref_col_type == SQLITE_FLOAT || ref_col_type == SQLITE_INTEGER);
  } else {
    CHECK_EQ(SQLITE_TEXT, ref_col_type);
  }
}

template <typename RESULT_SET>
bool definitelyHasNoRows(const RESULT_SET* rs) {
  return rs->definitelyHasNoRows();
}

template <>
bool definitelyHasNoRows<hdk::ResultSetTableToken>(
    const hdk::ResultSetTableToken* token) {
  return token->rowCount() == (size_t)0;
}

template <class RESULT_SET>
void compare_impl(SqliteConnector& connector,
                  bool use_row_iterator,
                  const RESULT_SET* omnisci_results,
                  const std::string& sqlite_query_string,
                  const ExecutorDeviceType device_type,
                  const bool timestamp_approx,
                  const bool is_arrow = false) {
  auto const errmsg = ExecutorDeviceType::CPU == device_type
                          ? "CPU: " + sqlite_query_string
                          : "GPU: " + sqlite_query_string;
  connector.query(sqlite_query_string);
  ASSERT_EQ(connector.getNumRows(), omnisci_results->rowCount()) << errmsg;
  const int num_rows{static_cast<int>(connector.getNumRows())};
  if (definitelyHasNoRows(omnisci_results)) {
    ASSERT_EQ(0, num_rows) << errmsg;
    return;
  }
  if (!num_rows) {
    return;
  }
  CHECK_EQ(connector.getNumCols(), omnisci_results->colCount()) << errmsg;
  const int num_cols{static_cast<int>(connector.getNumCols())};
  auto row_iterator = omnisci_results->rowIterator(true, true);
  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    std::vector<TargetValue> crt_row;
    if constexpr (std::is_same_v<RESULT_SET, ArrowResultSet>) {
      crt_row =
          use_row_iterator ? *row_iterator++ : omnisci_results->getNextRow(true, true);
    } else {
      crt_row = *row_iterator++;
    }
    CHECK(!crt_row.empty()) << errmsg;
    CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size()) << errmsg;
    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
      const auto ref_col_type = connector.columnTypes[col_idx];
      const auto omnisci_variant = crt_row[col_idx];
      const auto scalar_omnisci_variant = boost::get<ScalarTargetValue>(&omnisci_variant);
      CHECK(scalar_omnisci_variant) << errmsg;
      auto col_type = omnisci_results->colType(col_idx);
      checkTypeConsistency(ref_col_type, col_type);
      const bool ref_is_null = connector.isNull(row_idx, col_idx);
      switch (col_type->id()) {
        case hdk::ir::Type::kInteger: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_int_p);
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            ASSERT_EQ(inline_int_null_value(col_type), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<int64_t>(row_idx, col_idx);
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          }
          break;
        }
        case hdk::ir::Type::kExtDictionary:
        case hdk::ir::Type::kText:
        case hdk::ir::Type::kVarChar: {
          const auto omnisci_as_str_p =
              boost::get<NullableString>(scalar_omnisci_variant);
          ASSERT_NE(nullptr, omnisci_as_str_p) << errmsg;
          const auto omnisci_str_notnull = boost::get<std::string>(omnisci_as_str_p);
          const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
          if (omnisci_str_notnull) {
            const auto omnisci_val = *omnisci_str_notnull;
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          } else {
            // not null but no data, so val is empty string
            const auto omnisci_val = "";
            ASSERT_EQ(ref_val, omnisci_val) << errmsg;
          }
          break;
        }
        case hdk::ir::Type::kFloatingPoint:
        case hdk::ir::Type::kDecimal:
          if (!col_type->isFp32()) {
            const auto omnisci_as_double_p = boost::get<double>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_double_p) << errmsg;
            const auto omnisci_val = *omnisci_as_double_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_value(col_type->ctx().fp64()), omnisci_val)
                  << errmsg;
            } else {
              const auto ref_val = connector.getData<double>(row_idx, col_idx);
              if (!std::isinf(omnisci_val) || !std::isinf(ref_val) ||
                  ((omnisci_val < 0) ^ (ref_val < 0))) {
                ASSERT_NEAR(ref_val, omnisci_val, EPS * std::fabs(ref_val)) << errmsg;
              }
            }
          } else {
            const auto omnisci_as_float_p = boost::get<float>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_float_p) << errmsg;
            const auto omnisci_val = *omnisci_as_float_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_value(col_type), omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector.getData<float>(row_idx, col_idx);
              if (!std::isinf(omnisci_val) || !std::isinf(ref_val) ||
                  ((omnisci_val < 0) ^ (ref_val < 0))) {
                ASSERT_NEAR(ref_val, omnisci_val, EPS * std::fabs(ref_val)) << errmsg;
              }
            }
          }
          break;
        case hdk::ir::Type::kTimestamp:
        case hdk::ir::Type::kDate: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p);
          const auto omnisci_val = *omnisci_as_int_p;
          time_t nsec = 0;
          auto unit = dynamic_cast<const hdk::ir::DateTimeBaseType*>(col_type)->unit();
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_value(col_type), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            auto temp_val =
                dateTimeParseOptional<hdk::ir::Type::kTimestamp>(ref_val, unit);
            if (!temp_val) {
              temp_val = dateTimeParseOptional<hdk::ir::Type::kDate>(ref_val, unit);
            }
            CHECK(temp_val) << ref_val;
            nsec = temp_val.value();
            if (timestamp_approx) {
              // approximate result give 10 second lee way
              ASSERT_NEAR(*omnisci_as_int_p, nsec, hdk::ir::unitsPerSecond(unit) * 10);
            } else {
              struct tm tm_struct {
                0
              };
#ifdef _WIN32
              auto ret_code = gmtime_s(&tm_struct, &nsec);
              if (ret_code != 0)
                LOG(WARNING) << "gmtime_s returned errno value" << ret_code;
#else
              gmtime_r(&nsec, &tm_struct);
#endif
              if (is_arrow && col_type->isDate()) {
                if (device_type == ExecutorDeviceType::CPU) {
                  ASSERT_EQ(
                      *omnisci_as_int_p,
                      DateConverters::get_epoch_days_from_seconds(timegm(&tm_struct)))
                      << errmsg;
                } else {
                  ASSERT_EQ(*omnisci_as_int_p, timegm(&tm_struct) * kMilliSecsPerSec)
                      << errmsg;
                }
              } else {
                ASSERT_EQ(*omnisci_as_int_p,
                          unit > hdk::ir::TimeUnit::kSecond ? nsec : timegm(&tm_struct))
                    << errmsg;
              }
            }
          }
          break;
        }
        case hdk::ir::Type::kBoolean: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p) << errmsg;
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_value(col_type), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            if (ref_val == "t" || ref_val == "1") {
              ASSERT_EQ(1, *omnisci_as_int_p) << errmsg;
            } else {
              CHECK(ref_val == "f" || ref_val == "0") << errmsg;
              ASSERT_EQ(0, *omnisci_as_int_p) << errmsg;
            }
          }
          break;
        }
        case hdk::ir::Type::kTime: {
          const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
          CHECK(omnisci_as_int_p) << errmsg;
          const auto omnisci_val = *omnisci_as_int_p;
          if (ref_is_null) {
            CHECK_EQ(inline_int_null_value(col_type), omnisci_val) << errmsg;
          } else {
            const auto ref_val = connector.getData<std::string>(row_idx, col_idx);
            std::vector<std::string> time_tokens;
            boost::split(time_tokens, ref_val, boost::is_any_of(":"));
            ASSERT_EQ(size_t(3), time_tokens.size()) << errmsg;
            ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                          boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                          boost::lexical_cast<int64_t>(time_tokens[2]),
                      *omnisci_as_int_p)
                << errmsg;
          }
          break;
        }
        default:
          CHECK(false) << errmsg;
      }
    }
  }
}

}  // namespace

SQLiteComparator::SQLiteComparator(bool use_row_iterator)
    : sqlite_db_dir_("sqliteTestDB-"s + boost::filesystem::unique_path().string())
    , connector_(sqlite_db_dir_, "")
    , use_row_iterator_(use_row_iterator) {}

SQLiteComparator::~SQLiteComparator() {
  std::error_code err;
  std::filesystem::remove_all(sqlite_db_dir_, err);
}

void SQLiteComparator::compare(hdk::ResultSetTableTokenPtr omnisci_results,
                               const std::string& query_string,
                               const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               omnisci_results.get(),
               query_string,
               device_type,
               false);
}

void SQLiteComparator::compare_arrow_output(
    std::unique_ptr<ArrowResultSet>& arrow_omnisci_results,
    const std::string& sqlite_query_string,
    const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               arrow_omnisci_results.get(),
               sqlite_query_string,
               device_type,
               false,
               true);
}

// added to deal with time shift for now testing
void SQLiteComparator::compare_timstamp_approx(
    hdk::ResultSetTableTokenPtr omnisci_results,
    const std::string& query_string,
    const ExecutorDeviceType device_type) {
  compare_impl(connector_,
               use_row_iterator_,
               omnisci_results.get(),
               query_string,
               device_type,
               true);
}

}  // namespace TestHelpers
