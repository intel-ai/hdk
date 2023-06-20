/*
 * Copyright 2018 OmniSci, Inc.
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

#include "CardinalityEstimator.h"
#include "ErrorHandling.h"
#include "ExpressionRewrite.h"
#include "RelAlgExecutor.h"

size_t RelAlgExecutor::getNDVEstimation(const WorkUnit& work_unit,
                                        const int64_t range,
                                        const bool is_agg,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo) {
  const auto estimator_exe_unit = create_ndv_execution_unit(
      work_unit.exe_unit, schema_provider_.get(), config_, range);
  size_t one{1};
  ColumnCacheMap column_cache;
  try {
    const auto estimator_result =
        executor_->executeWorkUnit(one,
                                   is_agg,
                                   get_table_infos(work_unit.exe_unit, executor_),
                                   estimator_exe_unit,
                                   co,
                                   eo,
                                   false,
                                   data_provider_,
                                   column_cache);
    CHECK_EQ(estimator_result.size(), (size_t)1);
    return std::max(estimator_result.result(0)->getNDVEstimator(), size_t(1));
  } catch (const QueryExecutionError& e) {
    if (e.getErrorCode() == Executor::ERR_OUT_OF_TIME) {
      throw std::runtime_error("Cardinality estimation query ran out of time");
    }
    if (e.getErrorCode() == Executor::ERR_INTERRUPTED) {
      throw std::runtime_error("Cardinality estimation query has been interrupted");
    }
    throw std::runtime_error("Failed to run the cardinality estimation query: " +
                             getErrorMessageFromCode(e.getErrorCode()));
  }
  UNREACHABLE();
  return 0;
}

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit,
                                              SchemaProvider* schema_provider,
                                              const Config& config,
                                              const int64_t range) {
  bool use_large_estimator = range > config.exec.group_by.large_ndv_threshold;
  // Check if number of input rows is big enough to require large estimator.
  if (use_large_estimator) {
    auto outer_input_it = std::find_if(
        ra_exe_unit.input_descs.begin(),
        ra_exe_unit.input_descs.end(),
        [](const InputDescriptor& desc) { return desc.getNestLevel() == 0; });
    CHECK(outer_input_it != ra_exe_unit.input_descs.end());
    auto tinfo = schema_provider->getTableInfo(outer_input_it->getDatabaseId(),
                                               outer_input_it->getTableId());
    CHECK(tinfo);
    if (tinfo->row_count < config.exec.group_by.large_ndv_threshold) {
      LOG(INFO) << "Avoid large estimator because of the small input ("
                << tinfo->row_count << " rows).";
      use_large_estimator = false;
    }
  }
  size_t ndv_multiplier =
      use_large_estimator ? config.exec.group_by.large_ndv_multiplier : 1;
  LOG(INFO) << "Create NDV estimator execution unit for range = " << range
            << ". Chosen NDV multiplier is " << ndv_multiplier;
  return {
      ra_exe_unit.input_descs,
      ra_exe_unit.input_col_descs,
      ra_exe_unit.simple_quals,
      ra_exe_unit.quals,
      ra_exe_unit.join_quals,
      {},
      {},
      hdk::ir::makeExpr<hdk::ir::NDVEstimator>(ra_exe_unit.groupby_exprs, ndv_multiplier),
      SortInfo{{}, SortAlgorithm::Default, 0, 0},
      0,
      ra_exe_unit.query_plan_dag,
      ra_exe_unit.hash_table_build_plan_dag,
      ra_exe_unit.table_id_to_node_map,
      ra_exe_unit.union_all};
}

RelAlgExecutionUnit create_count_all_execution_unit(
    const RelAlgExecutionUnit& ra_exe_unit,
    hdk::ir::ExprPtr replacement_target) {
  return {ra_exe_unit.input_descs,
          ra_exe_unit.input_col_descs,
          ra_exe_unit.simple_quals,
          ra_exe_unit.quals,
          ra_exe_unit.join_quals,
          {},
          {replacement_target.get()},
          nullptr,
          SortInfo{{}, SortAlgorithm::Default, 0, 0},
          0,
          ra_exe_unit.query_plan_dag,
          ra_exe_unit.hash_table_build_plan_dag,
          ra_exe_unit.table_id_to_node_map,
          ra_exe_unit.union_all,
          ra_exe_unit.shuffle_fn,
          ra_exe_unit.partition_offsets_col,
          ra_exe_unit.partitioned_aggregation,
          ra_exe_unit.cost_model,
          {}};  // TODO(bagrorg): should we use costmodel here?
}

ResultSetPtr reduce_estimator_results(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  CHECK(dynamic_cast<const hdk::ir::NDVEstimator*>(ra_exe_unit.estimator.get()));
  const auto& result_set = results_per_device.front().first;
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto& next_result_set = results_per_device[i].first;
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    for (size_t off = 0; off < ra_exe_unit.estimator->getBufferSize(); ++off) {
      estimator_buffer[off] |= other_estimator_buffer[off];
    }
  }
  return std::move(result_set);
}
