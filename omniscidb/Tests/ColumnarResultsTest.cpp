/*
 * Copyright 2019 OmniSci, Inc.
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

#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "ResultSet/ResultSet.h"
#include "ResultSet/RowSetMemoryOwner.h"
#include "ResultSetRegistry/ColumnarResults.h"
#include "Shared/TargetInfo.h"
#include "Tests/ResultSetTestUtils.h"
#include "Tests/TestHelpers.h"

#include <gtest/gtest.h>

EXTERN extern bool g_is_test_env;

auto int8_type = hdk::ir::Context::defaultCtx().int8();
auto int16_type = hdk::ir::Context::defaultCtx().int16();
auto int32_type = hdk::ir::Context::defaultCtx().int32();
auto int64_type = hdk::ir::Context::defaultCtx().int64();
auto float_type = hdk::ir::Context::defaultCtx().fp32();
auto double_type = hdk::ir::Context::defaultCtx().fp64();

class ColumnarResultsTester : public ColumnarResults {
 public:
  ColumnarResultsTester(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const ResultSet& rows,
                        const size_t num_columns,
                        const std::vector<const hdk::ir::Type*> target_types,
                        Config& config,
                        const bool is_parallel_execution_enforced = false)
      : ColumnarResults(row_set_mem_owner,
                        rows,
                        num_columns,
                        target_types,
                        0,
                        config,
                        is_parallel_execution_enforced) {}

  template <typename ENTRY_TYPE>
  ENTRY_TYPE getEntryAt(const size_t row_idx, const size_t column_idx) const {
    CHECK_LT(column_idx, column_buffers_.size());
    CHECK_LT(row_idx, num_rows_);
    return reinterpret_cast<ENTRY_TYPE*>(column_buffers_[column_idx])[row_idx];
  }
};

template <>
float ColumnarResultsTester::getEntryAt<float>(const size_t row_idx,
                                               const size_t column_idx) const {
  CHECK_LT(column_idx, column_buffers_.size());
  CHECK_LT(row_idx, num_rows_);
  return reinterpret_cast<float*>(column_buffers_[column_idx])[row_idx];
}

template <>
double ColumnarResultsTester::getEntryAt<double>(const size_t row_idx,
                                                 const size_t column_idx) const {
  CHECK_LT(column_idx, column_buffers_.size());
  CHECK_LT(row_idx, num_rows_);
  return reinterpret_cast<double*>(column_buffers_[column_idx])[row_idx];
}

void test_columnar_conversion(const std::vector<TargetInfo>& target_infos,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const size_t non_empty_step_size,
                              const bool is_parallel_conversion = false) {
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      nullptr, Executor::getArenaBlockSize(), /*num_threads=*/1);
  ResultSet result_set(target_infos,
                       ExecutorDeviceType::CPU,
                       query_mem_desc,
                       row_set_mem_owner,
                       nullptr,
                       0,
                       0);

  // fill the storage
  const auto storage = result_set.allocateStorage();
  EvenNumberGenerator generator;
  fill_storage_buffer(storage->getUnderlyingBuffer(),
                      target_infos,
                      query_mem_desc,
                      generator,
                      0,
                      non_empty_step_size);

  // Columnar Conversion:
  std::vector<const hdk::ir::Type*> col_types;
  for (size_t i = 0; i < result_set.colCount(); ++i) {
    col_types.push_back(result_set.colType(i)->canonicalize());
  }
  Config config;
  ColumnarResultsTester columnar_results(row_set_mem_owner,
                                         result_set,
                                         col_types.size(),
                                         col_types,
                                         config,
                                         is_parallel_conversion);

  // Validate the results:
  for (size_t rs_row_idx = 0, cr_row_idx = 0; rs_row_idx < query_mem_desc.getEntryCount();
       rs_row_idx++) {
    if (result_set.isRowAtEmpty(rs_row_idx)) {
      // empty entries should be filtered out for conversion:
      continue;
    }
    const auto row = result_set.getRowAt(rs_row_idx);
    if (row.empty()) {
      break;
    }
    CHECK_EQ(target_infos.size(), row.size());
    for (size_t target_idx = 0; target_idx < target_infos.size(); ++target_idx) {
      const auto& target_info = target_infos[target_idx];
      auto type = target_info.agg_kind == hdk::ir::AggType::kAvg
                      ? target_info.type->ctx().fp64()
                      : target_info.type;
      switch (type->id()) {
        case hdk::ir::Type::kInteger:
          switch (type->size()) {
            case 8: {
              const auto ival_result_set = v<int64_t>(row[target_idx]);
              const auto ival_converted = static_cast<int64_t>(
                  columnar_results.getEntryAt<int64_t>(cr_row_idx, target_idx));
              ASSERT_EQ(ival_converted, ival_result_set);
              break;
            }
            case 4: {
              const auto ival_result_set = v<int64_t>(row[target_idx]);
              const auto ival_converted = static_cast<int64_t>(
                  columnar_results.getEntryAt<int32_t>(cr_row_idx, target_idx));
              ASSERT_EQ(ival_converted, ival_result_set);
              break;
            }
            case 2: {
              const auto ival_result_set = v<int64_t>(row[target_idx]);
              const auto ival_converted = static_cast<int64_t>(
                  columnar_results.getEntryAt<int16_t>(cr_row_idx, target_idx));
              ASSERT_EQ(ival_result_set, ival_converted);
              break;
            }
            case 1: {
              const auto ival_result_set = v<int64_t>(row[target_idx]);
              const auto ival_converted = static_cast<int64_t>(
                  columnar_results.getEntryAt<int8_t>(cr_row_idx, target_idx));
              ASSERT_EQ(ival_converted, ival_result_set);
              break;
            }
            default:
              UNREACHABLE() << "Invalid type info encountered.";
          }
          break;
        case hdk::ir::Type::kFloatingPoint:
          switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
            case hdk::ir::FloatingPointType::kFloat: {
              const auto fval_result_set = v<float>(row[target_idx]);
              const auto fval_converted =
                  columnar_results.getEntryAt<float>(cr_row_idx, target_idx);
              ASSERT_FLOAT_EQ(fval_result_set, fval_converted);
              break;
            }
            case hdk::ir::FloatingPointType::kDouble: {
              const auto dval_result_set = v<double>(row[target_idx]);
              const auto dval_converted =
                  columnar_results.getEntryAt<double>(cr_row_idx, target_idx);
              ASSERT_FLOAT_EQ(dval_result_set, dval_converted);
              break;
            }
            default:
              UNREACHABLE() << "Invalid type info encountered.";
          }
          break;
        default:
          UNREACHABLE() << "Invalid type info encountered.";
      }
    }
    cr_row_idx++;
  }
}

TEST(Construct, Empty) {
  std::vector<TargetInfo> target_infos;
  std::vector<const hdk::ir::Type*> types;
  QueryMemoryDescriptor query_mem_desc;
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      nullptr, Executor::getArenaBlockSize(), /*num_threads=*/1);
  ResultSet result_set(target_infos,
                       ExecutorDeviceType::CPU,
                       query_mem_desc,
                       row_set_mem_owner,
                       nullptr,
                       0,
                       0);
  Config config;
  ColumnarResultsTester columnar_results(
      row_set_mem_owner, result_set, types.size(), types, config);
}

// Projections:
// TODO(Saman): add tests for Projections

// Perfect Hash:
TEST(PerfectHashRowWise, OneCol_64Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kSum,
       hdk::ir::AggType::kSum,
       hdk::ir::AggType::kCount,
       hdk::ir::AggType::kMin,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMin},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kCount,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMin},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        double_type,
                                        int64_type},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kSum,
       hdk::ir::AggType::kSum,
       hdk::ir::AggType::kCount,
       hdk::ir::AggType::kMin,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMin},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 3, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kCount,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMin},
                                       {int64_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        int64_type},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 3, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        double_type,
                                        int64_type,
                                        float_type,
                                        double_type,
                                        double_type},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        int16_type,
                                        int64_type,
                                        float_type,
                                        int32_type,
                                        double_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {double_type,
                                        double_type,
                                        float_type,
                                        int64_type,
                                        double_type,
                                        int32_type,
                                        int16_type,
                                        int8_type},
                                       {double_type,
                                        double_type,
                                        float_type,
                                        int64_type,
                                        float_type,
                                        int32_type,
                                        int16_type,
                                        int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kCount,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMin},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        double_type,
                                        int64_type},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kSum,
       hdk::ir::AggType::kSum,
       hdk::ir::AggType::kCount,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMin},
      {int64_type, int64_type, int64_type, int64_type, int64_type},
      {int64_type, int64_type, int64_type, int64_type, int64_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {double_type,
                                        float_type,
                                        double_type,
                                        int64_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        int8_type},
                                       {double_type,
                                        float_type,
                                        int32_type,
                                        int64_type,
                                        int32_type,
                                        int16_type,
                                        int16_type,
                                        int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_32Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {double_type,
                                        float_type,
                                        double_type,
                                        double_type,
                                        int64_type,
                                        int32_type,
                                        int16_type,
                                        int8_type},
                                       {double_type,
                                        float_type,
                                        int64_type,
                                        int8_type,
                                        int64_type,
                                        int32_type,
                                        int16_type,
                                        int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_32Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_16Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax},
                                       {double_type,
                                        double_type,
                                        float_type,
                                        int64_type,
                                        int32_type,
                                        int16_type,
                                        double_type,
                                        int8_type},
                                       {double_type,
                                        int32_type,
                                        float_type,
                                        int64_type,
                                        int32_type,
                                        int16_type,
                                        int64_type,
                                        int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_16Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_8Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax},
                                       {double_type,
                                        float_type,
                                        int64_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type,
                                        int8_type},
                                       {double_type,
                                        float_type,
                                        int64_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        int16_type,
                                        int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_8Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type},
      {double_type, float_type, int64_type, int32_type, int16_type, int8_type});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

// Multi-column perfect hash:
TEST(PerfectHashRowWise, TwoCol_64_64_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kSum,
       hdk::ir::AggType::kSum,
       hdk::ir::AggType::kCount,
       hdk::ir::AggType::kMin,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMin},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kCount,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMin},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        double_type,
                                        int64_type},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        double_type,
                                        int64_type,
                                        float_type,
                                        double_type,
                                        double_type},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        int16_type,
                                        int64_type,
                                        float_type,
                                        int32_type,
                                        double_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_TwoCol_64_64_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kSum,
       hdk::ir::AggType::kSum,
       hdk::ir::AggType::kCount,
       hdk::ir::AggType::kMin,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMin},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type},
      {int64_type, int64_type, int64_type, double_type, int64_type, int64_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kSum,
                                        hdk::ir::AggType::kCount,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMin},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        double_type,
                                        int64_type,
                                        double_type,
                                        int64_type},
                                       {int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type,
                                        int64_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type},
      {int8_type, int16_type, int32_type, int64_type, float_type, double_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        double_type,
                                        int64_type,
                                        float_type,
                                        double_type,
                                        double_type},
                                       {int8_type,
                                        int16_type,
                                        int32_type,
                                        int16_type,
                                        int64_type,
                                        float_type,
                                        int32_type,
                                        double_type});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

// Baseline Hash:
TEST(BaselineHashRowWise, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_64_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_32_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax,
       hdk::ir::AggType::kMax},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type},
      {float_type, int64_type, int8_type, int32_type, int16_type, double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_32_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kAvg,
                                        hdk::ir::AggType::kMax,
                                        hdk::ir::AggType::kMax},
                                       {float_type,
                                        double_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        double_type,
                                        int16_type,
                                        double_type},
                                       {float_type,
                                        int8_type,
                                        int64_type,
                                        int8_type,
                                        int32_type,
                                        int32_type,
                                        int16_type,
                                        double_type});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

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
