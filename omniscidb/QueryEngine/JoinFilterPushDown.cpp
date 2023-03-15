/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "JoinFilterPushDown.h"
#include "IR/ExprCollector.h"
#include "IR/ExprRewriter.h"
#include "RelAlgExecutor.h"

namespace {

class BindFilterToOutermostVisitor : public hdk::ir::ExprRewriter {
  hdk::ir::ExprPtr visitColumnVar(const hdk::ir::ColumnVar* col_var) override {
    return hdk::ir::makeExpr<hdk::ir::ColumnVar>(col_var->columnInfo(), 0);
  }
};

class InputColumnsCollector
    : public hdk::ir::ExprCollector<std::unordered_set<InputColDescriptor>,
                                    InputColumnsCollector> {
 protected:
  void visitColumnVar(const hdk::ir::ColumnVar* col_var) override {
    result_.insert(InputColDescriptor(col_var->columnInfo(), 0));
  }
};

}  // namespace

/**
 * Given a set of filter expressions for a table, it launches a new COUNT query to
 * compute the number of passing rows, and then generates a set of statistics
 * related to those filters.
 * Later, these stats are used to decide whether
 * a filter should be pushed down or not.
 */
FilterSelectivity RelAlgExecutor::getFilterSelectivity(
    const std::vector<hdk::ir::ExprPtr>& filter_expressions,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  InputColumnsCollector input_columns_collector;
  std::list<hdk::ir::ExprPtr> quals;
  BindFilterToOutermostVisitor bind_filter_to_outermost;
  for (const auto& filter_expr : filter_expressions) {
    input_columns_collector.visit(filter_expr.get());
    quals.push_back(bind_filter_to_outermost.visit(filter_expr.get()));
  }
  auto& input_column_descriptors = input_columns_collector.result();
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  for (const auto& input_col_desc : input_column_descriptors) {
    if (input_descs.empty()) {
      input_descs.push_back(input_col_desc.getScanDesc());
    } else {
      CHECK(input_col_desc.getScanDesc() == input_descs.front());
    }
    input_col_descs.push_back(std::make_shared<const InputColDescriptor>(input_col_desc));
  }
  const auto count_expr = hdk::ir::makeExpr<hdk::ir::AggExpr>(
      hdk::ir::Context::defaultCtx().integer(config_.exec.group_by.bigint_count ? 8 : 4),
      hdk::ir::AggType::kCount,
      nullptr,
      false,
      nullptr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {},
                                  quals,
                                  {},
                                  {},
                                  {count_expr.get()},
                                  nullptr,
                                  {{}, SortAlgorithm::Default, 0, 0},
                                  0};
  size_t one{1};
  TemporaryTable filtered_result;
  const auto table_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), table_infos.size());
  const size_t total_rows_upper_bound = table_infos.front().info.getNumTuplesUpperBound();
  try {
    ColumnCacheMap column_cache;
    filtered_result = executor_->executeWorkUnit(
        one, true, table_infos, ra_exe_unit, co, eo, false, data_provider_, column_cache);
  } catch (...) {
    return {false, 1.0, 0};
  }
  CHECK_EQ(filtered_result.getFragCount(), 1);
  const auto count_row = filtered_result[0]->getNextRow(false, false);
  CHECK_EQ(size_t(1), count_row.size());
  const auto& count_tv = count_row.front();
  const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
  CHECK(count_scalar_tv);
  const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
  CHECK(count_ptr);
  const auto rows_passing = *count_ptr;
  const auto rows_total = std::max(total_rows_upper_bound, size_t(1));
  return {true, static_cast<float>(rows_passing) / rows_total, total_rows_upper_bound};
}

/**
 * Goes through all candidate filters and evaluate whether they pass the selectivity
 * criteria or not.
 */
std::vector<PushedDownFilterInfo> RelAlgExecutor::selectFiltersToBePushedDown(
    const RelAlgExecutor::WorkUnit& work_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  const auto all_push_down_candidates =
      find_push_down_filters(work_unit.exe_unit,
                             work_unit.input_permutation,
                             work_unit.left_deep_join_input_sizes);
  std::vector<PushedDownFilterInfo> selective_push_down_candidates;
  const auto ti = get_table_infos(work_unit.exe_unit.input_descs, executor_);
  if (to_gather_info_for_filter_selectivity(ti)) {
    for (const auto& candidate : all_push_down_candidates) {
      const auto selectivity = getFilterSelectivity(candidate.filter_expressions, co, eo);
      if (selectivity.is_valid &&
          selectivity.isFilterSelectiveEnough(config_.opts.filter_pushdown)) {
        selective_push_down_candidates.push_back(candidate);
      }
    }
  }
  return selective_push_down_candidates;
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryWithFilterPushDown(
    const RaExecutionSequence& seq,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const int64_t queue_time_ms) {
  // we currently do not fully support filter push down with
  // multi-step execution and/or with subqueries
  // TODO(Saman): add proper caching to enable filter push down for all cases
  const auto& subqueries = getSubqueries();
  if (seq.size() > 1 || !subqueries.empty()) {
    if (eo.just_calcite_explain) {
      return ExecutionResult(std::vector<PushedDownFilterInfo>{},
                             eo.find_push_down_candidates);
    }

    const ExecutionOptions eo_modified = [&]() {
      ExecutionOptions copy = eo;
      copy.find_push_down_candidates = false;
      copy.outer_fragment_indices = {};
      return copy;
    }();

    // Dispatch the subqueries first
    for (auto& subquery : subqueries) {
      // Execute the subquery and cache the result.
      RelAlgExecutor ra_executor(executor_, schema_provider_, data_provider_);
      const auto subquery_ra = subquery->node();
      CHECK(subquery_ra);
      RaExecutionSequence subquery_seq(subquery_ra);
      auto result = ra_executor.executeRelAlgSeq(subquery_seq, co, eo_modified, 0);
      auto shared_result = std::make_shared<ExecutionResult>(std::move(result));
      subquery_ra->setResult(shared_result);
    }
    return executeRelAlgSeq(seq, co, eo_modified, queue_time_ms);
  }
  // else
  return executeRelAlgSeq(seq, co, eo, queue_time_ms);
}
/**
 * The main purpose of this function is to prevent going through extra overhead of
 * computing required statistics for finding the right candidates and then the actual
 * push-down, unless the problem is large enough that such effort is potentially helpful.
 */
bool to_gather_info_for_filter_selectivity(
    const std::vector<InputTableInfo>& table_infos) {
  if (table_infos.size() < 2) {
    return false;
  }
  // we currently do not support filter push down when there is a self-join involved:
  // TODO(Saman): prevent Calcite from optimizing self-joins to remove this exclusion
  std::unordered_set<int> table_ids;
  for (auto ti : table_infos) {
    if (table_ids.find(ti.table_id) == table_ids.end()) {
      table_ids.insert(ti.table_id);
    } else {
      // a self-join is involved
      return false;
    }
  }
  // TODO(Saman): add some extra heuristics to avoid preflight count and push down if it
  // is not going to be helpful.
  return true;
}

/**
 * Go through all tables involved in the relational algebra plan, and select potential
 * candidates to be pushed down by calcite. For each filter we store a set of
 * intermediate indices (previous, current, and next table) based on the column
 * indices in their query string.
 */
std::vector<PushedDownFilterInfo> find_push_down_filters(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<size_t>& input_permutation,
    const std::vector<size_t>& left_deep_join_input_sizes) {
  std::vector<PushedDownFilterInfo> result;
  if (left_deep_join_input_sizes.empty()) {
    return result;
  }
  std::vector<size_t> input_size_prefix_sums(left_deep_join_input_sizes.size());
  std::partial_sum(left_deep_join_input_sizes.begin(),
                   left_deep_join_input_sizes.end(),
                   input_size_prefix_sums.begin());
  std::vector<int> to_original_rte_idx(ra_exe_unit.input_descs.size(),
                                       ra_exe_unit.input_descs.size());
  if (!input_permutation.empty()) {
    CHECK_EQ(to_original_rte_idx.size(), input_permutation.size());
    for (size_t i = 0; i < input_permutation.size(); ++i) {
      CHECK_LT(input_permutation[i], to_original_rte_idx.size());
      CHECK_EQ(static_cast<size_t>(to_original_rte_idx[input_permutation[i]]),
               to_original_rte_idx.size());
      to_original_rte_idx[input_permutation[i]] = i;
    }
  } else {
    std::iota(to_original_rte_idx.begin(), to_original_rte_idx.end(), 0);
  }
  std::unordered_map<int, std::vector<hdk::ir::ExprPtr>> filters_per_nesting_level;
  for (const auto& level_conditions : ra_exe_unit.join_quals) {
    for (const auto& cond : level_conditions.quals) {
      const auto rte_indices = AllRangeTableIndexCollector::collect(cond.get());
      if (rte_indices.size() > 1) {
        continue;
      }
      const int rte_idx = (!rte_indices.empty()) ? *rte_indices.cbegin() : 0;
      if (!rte_idx) {
        continue;
      }
      CHECK_GE(rte_idx, 0);
      CHECK_LT(static_cast<size_t>(rte_idx), to_original_rte_idx.size());
      filters_per_nesting_level[to_original_rte_idx[rte_idx]].push_back(cond);
    }
  }
  for (const auto& kv : filters_per_nesting_level) {
    CHECK_GE(kv.first, 0);
    CHECK_LT(static_cast<size_t>(kv.first), input_size_prefix_sums.size());
    size_t input_prev = (kv.first > 1) ? input_size_prefix_sums[kv.first - 2] : 0;
    size_t input_start = kv.first ? input_size_prefix_sums[kv.first - 1] : 0;
    size_t input_next = input_size_prefix_sums[kv.first];
    result.emplace_back(
        PushedDownFilterInfo{kv.second, input_prev, input_start, input_next});
  }
  return result;
}
