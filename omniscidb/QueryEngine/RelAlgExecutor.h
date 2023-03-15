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

#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "DataProvider/DataProvider.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/QueryExecutionSequence.h"
#include "QueryEngine/QueryRewrite.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgSchemaProvider.h"
#include "QueryEngine/SpeculativeTopN.h"
#include "QueryEngine/StreamingTopN.h"
#include "Shared/scope.h"

#include <ctime>
#include <sstream>

enum class MergeType { Union, Reduce };

struct QueryStepExecutionResult {
  ExecutionResult result;
  const MergeType merge_type;
  const unsigned node_id;
  bool is_outermost_query;
};

class RelAlgExecutor {
 public:
  using TargetInfoList = std::vector<TargetInfo>;

  RelAlgExecutor(Executor* executor,
                 SchemaProviderPtr schema_provider,
                 DataProvider* data_provider);

  RelAlgExecutor(Executor* executor,
                 SchemaProviderPtr schema_provider,
                 DataProvider* data_provider,
                 std::unique_ptr<hdk::ir::QueryDag> query_dag);

  size_t getOuterFragmentCount(const CompilationOptions& co, const ExecutionOptions& eo);

  ExecutionResult executeRelAlgQuery(const CompilationOptions& co,
                                     const ExecutionOptions& eo,
                                     const bool just_explain_plan);

  // does preparational stuff and compiles kernels
  void prepareStreamingExecution(const CompilationOptions& co,
                                 const ExecutionOptions& eo);

  // process batch
  ResultSetPtr runOnBatch(const FragmentsPerTable& fragments);

  // when the last batch has been send, invoke this function.
  ResultSetPtr finishStreamingExecution();

  ExecutionResult executeRelAlgQueryWithFilterPushDown(const RaExecutionSequence& seq,
                                                       const CompilationOptions& co,
                                                       const ExecutionOptions& eo,
                                                       const int64_t queue_time_ms);

  void prepareLeafExecution(
      const AggregatedColRange& agg_col_range,
      const StringDictionaryGenerations& string_dictionary_generations,
      const TableGenerations& table_generations);

  ExecutionResult executeRelAlgSeq(const RaExecutionSequence& seq,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   const int64_t queue_time_ms,
                                   const bool with_existing_temp_tables = false);
  std::shared_ptr<const ExecutionResult> execute(
      const hdk::QueryExecutionSequence& seq,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      const int64_t queue_time_ms,
      const bool with_existing_temp_tables = false);

  ExecutionResult executeRelAlgSubSeq(const RaExecutionSequence& seq,
                                      const std::pair<size_t, size_t> interval,
                                      const CompilationOptions& co,
                                      const ExecutionOptions& eo,
                                      const int64_t queue_time_ms);

  QueryStepExecutionResult executeRelAlgQuerySingleStep(const RaExecutionSequence& seq,
                                                        const size_t step_idx,
                                                        const CompilationOptions& co,
                                                        const ExecutionOptions& eo);

  const hdk::ir::Node* getRootNode() const {
    CHECK(query_dag_);
    return query_dag_->getRootNode();
  }

  std::shared_ptr<const hdk::ir::Node> getRootNodeShPtr() const {
    CHECK(query_dag_);
    return query_dag_->getRootNodeShPtr();
  }

  std::pair<std::vector<unsigned>, std::unordered_map<unsigned, JoinQualsPerNestingLevel>>
  getJoinInfo(const hdk::ir::Node* root_node);

  std::shared_ptr<RelAlgTranslator> getRelAlgTranslator(const hdk::ir::Node* root_node);

  const std::vector<std::shared_ptr<const hdk::ir::ScalarSubquery>>& getSubqueries()
      const noexcept {
    CHECK(query_dag_);
    return query_dag_->getSubqueries();
  };

  AggregatedColRange computeColRangesCache();
  StringDictionaryGenerations computeStringDictionaryGenerations();
  TableGenerations computeTableGenerations();

  Executor* getExecutor() const;

  void cleanupPostExecution();

  static std::string getErrorMessageFromCode(const int32_t error_code);

  void executePostExecutionCallback();

  static const SpeculativeTopNBlacklist& speculativeTopNBlacklist() {
    return speculative_topn_blacklist_;
  }

 private:
  ExecutionResult executeRelAlgQueryNoRetry(const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            const bool just_explain_plan);

  void executeRelAlgStep(const RaExecutionSequence& seq,
                         const size_t step_idx,
                         const CompilationOptions&,
                         const ExecutionOptions&,
                         const int64_t queue_time_ms);
  void executeStep(const hdk::ir::Node* step_root,
                   const CompilationOptions& co,
                   const ExecutionOptions& eo,
                   const int64_t queue_time_ms);
  ExecutionResult executeStep(const hdk::ir::Node* step_root,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo,
                              const int64_t queue_time_ms,
                              bool allow_speculative_sort);

  ExecutionResult executeCompound(const hdk::ir::Compound*,
                                  const CompilationOptions&,
                                  const ExecutionOptions&,
                                  const int64_t queue_time_ms);

  ExecutionResult executeAggregate(const hdk::ir::Aggregate* aggregate,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   const int64_t queue_time_ms);

  ExecutionResult executeProject(const hdk::ir::Project*,
                                 const CompilationOptions&,
                                 const ExecutionOptions&,
                                 const int64_t queue_time_ms,
                                 const std::optional<size_t> previous_count);

  // Computes the window function results to be used by the query.
  void computeWindow(const RelAlgExecutionUnit& ra_exe_unit,
                     const CompilationOptions& co,
                     const ExecutionOptions& eo,
                     ColumnCacheMap& column_cache_map,
                     const int64_t queue_time_ms);

  // Creates the window context for the given window function.
  std::unique_ptr<WindowFunctionContext> createWindowFunctionContext(
      const hdk::ir::WindowFunction* window_func,
      const std::shared_ptr<const hdk::ir::BinOper>& partition_key_cond,
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& query_infos,
      const CompilationOptions& co,
      ColumnCacheMap& column_cache_map,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ExecutionResult executeFilter(const hdk::ir::Filter*,
                                const CompilationOptions&,
                                const ExecutionOptions&,
                                const int64_t queue_time_ms);

  ExecutionResult executeSort(const hdk::ir::Sort*,
                              const CompilationOptions&,
                              const ExecutionOptions&,
                              const int64_t queue_time_ms);

  ExecutionResult executeLogicalValues(const hdk::ir::LogicalValues*,
                                       const ExecutionOptions&);

  ExecutionResult executeUnion(const hdk::ir::LogicalUnion*,
                               const RaExecutionSequence&,
                               const CompilationOptions&,
                               const ExecutionOptions&,
                               const int64_t queue_time_ms);

  // TODO(alex): just move max_groups_buffer_entry_guess to RelAlgExecutionUnit once
  //             we deprecate the plan-based executor paths and remove WorkUnit
  struct WorkUnit {
    RelAlgExecutionUnit exe_unit;
    const hdk::ir::Node* body;
    const size_t max_groups_buffer_entry_guess;
    std::unique_ptr<QueryRewriter> query_rewriter;
    const std::vector<size_t> input_permutation;
    const std::vector<size_t> left_deep_join_input_sizes;
  };

  WorkUnit createWorkUnitForStreaming(const hdk::ir::Node* body,
                                      const CompilationOptions& co,
                                      const ExecutionOptions& eo);

  WorkUnit createSortInputWorkUnit(const hdk::ir::Sort*, const ExecutionOptions& eo);

  ExecutionResult executeWorkUnit(
      const WorkUnit& work_unit,
      const std::vector<TargetMetaInfo>& targets_meta,
      const bool is_agg,
      const CompilationOptions& co_in,
      const ExecutionOptions& eo_in,
      const int64_t queue_time_ms,
      const std::optional<size_t> previous_count = std::nullopt);

  size_t getNDVEstimation(const WorkUnit& work_unit,
                          const int64_t range,
                          const bool is_agg,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo);

  std::optional<size_t> getFilteredCountAll(const WorkUnit& work_unit,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo);

  FilterSelectivity getFilterSelectivity(
      const std::vector<hdk::ir::ExprPtr>& filter_expressions,
      const CompilationOptions& co,
      const ExecutionOptions& eo);

  std::vector<PushedDownFilterInfo> selectFiltersToBePushedDown(
      const RelAlgExecutor::WorkUnit& work_unit,
      const CompilationOptions& co,
      const ExecutionOptions& eo);

  bool isRowidLookup(const WorkUnit& work_unit);

  ExecutionResult handleOutOfMemoryRetry(const RelAlgExecutor::WorkUnit& work_unit,
                                         const std::vector<TargetMetaInfo>& targets_meta,
                                         const bool is_agg,
                                         const CompilationOptions& co,
                                         const ExecutionOptions& eo,
                                         const bool was_multifrag_kernel_launch,
                                         const int64_t queue_time_ms);

  // Allows an out of memory error through if CPU retry is enabled. Otherwise, throws an
  // appropriate exception corresponding to the query error code.
  void handlePersistentError(const int32_t error_code);

  WorkUnit createWorkUnit(const hdk::ir::Node*,
                          const SortInfo&,
                          const ExecutionOptions& eo);
  WorkUnit createWorkUnit(const hdk::ir::Node* node,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo,
                          bool allow_speculative_sort);

  WorkUnit createCompoundWorkUnit(const hdk::ir::Compound*,
                                  const SortInfo&,
                                  const ExecutionOptions& eo);

  WorkUnit createAggregateWorkUnit(const hdk::ir::Aggregate*,
                                   const SortInfo&,
                                   const bool just_explain);

  WorkUnit createProjectWorkUnit(const hdk::ir::Project*,
                                 const SortInfo&,
                                 const ExecutionOptions& eo);

  WorkUnit createFilterWorkUnit(const hdk::ir::Filter*,
                                const SortInfo&,
                                const bool just_explain);

  WorkUnit createUnionWorkUnit(const hdk::ir::LogicalUnion*,
                               const SortInfo&,
                               const ExecutionOptions& eo);

  void addTemporaryTable(const int table_id, const ResultSetPtr& result) {
    CHECK_LT(size_t(0), result->colCount());
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, result);
    CHECK(it_ok.second);
  }

  void addTemporaryTable(const int table_id, const TemporaryTable& table) {
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, table);
    CHECK(it_ok.second);
  }

  void eraseFromTemporaryTables(const int table_id) { temporary_tables_.erase(table_id); }

  void handleNop(RaExecutionDesc& ed);

  std::unordered_map<unsigned, JoinQualsPerNestingLevel>& getLeftDeepJoinTreesInfo() {
    return left_deep_join_info_;
  }

  JoinQualsPerNestingLevel translateLeftDeepJoinFilter(
      const hdk::ir::LeftDeepInnerJoin* join,
      const std::vector<InputDescriptor>& input_descs,
      const std::unordered_map<const hdk::ir::Node*, int>& input_to_nest_level,
      const bool just_explain);

  // Transform the provided `join_condition` to conjunctive form, find composite
  // key opportunities and finally translate it to an Analyzer expression.
  std::list<hdk::ir::ExprPtr> makeJoinQuals(
      const hdk::ir::Expr* join_condition_expr,
      const std::vector<JoinType>& join_types,
      const std::unordered_map<const hdk::ir::Node*, int>& input_to_nest_level,
      const bool just_explain) const;

  Executor* executor_;
  std::unique_ptr<hdk::ir::QueryDag> query_dag_;
  std::shared_ptr<SchemaProvider> schema_provider_;
  DataProvider* data_provider_;
  const Config& config_;
  TemporaryTables temporary_tables_;
  time_t now_;
  std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_join_info_;
  std::vector<hdk::ir::ExprPtr> target_exprs_owned_;  // TODO(alex): remove
  int64_t queue_time_ms_;
  static SpeculativeTopNBlacklist speculative_topn_blacklist_;

  std::optional<std::function<void()>> post_execution_callback_;

  std::shared_ptr<StreamExecutionContext> stream_execution_context_;

  friend class PendingExecutionClosure;
};

hdk::ir::ExprPtr set_transient_dict(const hdk::ir::ExprPtr expr);
hdk::ir::ExprPtr translate(const hdk::ir::Expr* expr,
                           const RelAlgTranslator& translator,
                           ::ExecutorType executor_type);
bool first_oe_is_desc(const std::list<hdk::ir::OrderEntry>& order_entries);
std::list<hdk::ir::OrderEntry> get_order_entries(const hdk::ir::Sort* sort);
std::vector<JoinType> left_deep_join_types(
    const hdk::ir::LeftDeepInnerJoin* left_deep_join);
std::vector<size_t> get_left_deep_join_input_sizes(
    const hdk::ir::LeftDeepInnerJoin* left_deep_join);
hdk::ir::ExprPtr get_bitwise_equals_conjunction(const hdk::ir::Expr* expr);
hdk::ir::ExprPtr reverse_logical_distribution(const hdk::ir::ExprPtr& expr);

const hdk::ir::Type* canonicalTypeForExpr(const hdk::ir::Expr& expr);

#endif  // QUERYENGINE_RELALGEXECUTOR_H
