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

#include "RelAlgExecutor.h"
#include "DataMgr/DataMgr.h"
#include "IR/TypeUtils.h"
#include "QueryBuilder/QueryBuilder.h"
#include "QueryEngine/CalciteDeserializerUtils.h"
#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExtensionFunctionsBinding.h"
#include "QueryEngine/ExternalExecutor.h"
#include "QueryEngine/FromTableReordering.h"
#include "QueryEngine/MemoryLayoutBuilder.h"
#include "QueryEngine/QueryPhysicalInputsCollector.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/RelAlgVisitor.h"
#include "QueryEngine/ResultSetBuilder.h"
#include "QueryEngine/ResultSetSort.h"
#include "QueryEngine/UnnestedVarsCollector.h"
#include "QueryEngine/WindowContext.h"
#include "QueryEngine/WorkUnitBuilder.h"
#include "QueryOptimizer/CanonicalizeQuery.h"
#include "ResultSet/ColRangeInfo.h"
#include "ResultSet/HyperLogLog.h"
#include "ResultSetRegistry/ResultSetRegistry.h"
#include "SchemaMgr/SchemaMgr.h"
#include "SessionInfo.h"
#include "Shared/MathUtils.h"
#include "Shared/funcannotations.h"
#include "Shared/measure.h"
#include "Shared/misc.h"

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/make_unique.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

using namespace std::string_literals;

size_t g_estimator_failure_max_groupby_size{256000000};
bool g_columnar_large_projections{true};
size_t g_columnar_large_projections_threshold{1000000};

EXTERN extern bool g_enable_table_functions;

namespace {

bool is_projection(const RelAlgExecutionUnit& ra_exe_unit) {
  return ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front();
}

bool should_output_columnar(const RelAlgExecutionUnit& ra_exe_unit) {
  if (!is_projection(ra_exe_unit)) {
    return false;
  }
  if (!ra_exe_unit.sort_info.order_entries.empty()) {
    // disable output columnar when we have top-sort node query
    return false;
  }
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    // We don't currently support varlen columnar projections, so
    // return false if we find one
    if (target_expr->type()->isString() || target_expr->type()->isArray()) {
      return false;
    }
  }

  return ra_exe_unit.scan_limit >= g_columnar_large_projections_threshold;
}

bool is_extracted_dag_valid(ExtractedPlanDag& dag) {
  return !dag.contain_not_supported_rel_node &&
         dag.extracted_dag.compare(EMPTY_QUERY_PLAN) != 0;
}

}  // namespace

RelAlgExecutor::RelAlgExecutor(Executor* executor, SchemaProviderPtr schema_provider)
    : executor_(executor)
    , schema_provider_(schema_provider)
    , data_provider_(executor->getDataMgr()->getDataProvider())
    , config_(executor_->getConfig())
    , now_(0)
    , queue_time_ms_(0) {
  rs_registry_ = hdk::ResultSetRegistry::getOrCreate(executor->getDataMgr(),
                                                     executor->getConfigPtr());
}

RelAlgExecutor::RelAlgExecutor(Executor* executor,
                               SchemaProviderPtr schema_provider,
                               std::unique_ptr<hdk::ir::QueryDag> query_dag)
    : executor_(executor)
    , query_dag_(std::move(query_dag))
    , schema_provider_(schema_provider)
    , data_provider_(executor->getDataMgr()->getDataProvider())
    , config_(executor_->getConfig())
    , now_(0)
    , queue_time_ms_(0) {
  rs_registry_ = hdk::ResultSetRegistry::getOrCreate(executor->getDataMgr(),
                                                     executor->getConfigPtr());

  // Add ResultSetRegistry to the schema provider by wrapping the current provider
  // and the registry in SchemaMgr.
  // TODO: In the future we expect pre-initialized registry and passed schema provider
  // to cover it.
  auto db_ids = schema_provider->listDatabases();
  if (std::find(db_ids.begin(), db_ids.end(), hdk::ResultSetRegistry::DB_ID) ==
      db_ids.end()) {
    schema_provider_ =
        mergeProviders(std::vector<SchemaProviderPtr>({schema_provider, rs_registry_}));
  }

  hdk::ir::canonicalizeQuery(*query_dag_);
}

RelAlgExecutor::~RelAlgExecutor() {
  // We don't need temporary tables anymore. On desctruction we are going to lose
  // all tokens and have ResultSets removed from the ResultSetRegistry. But some
  // zero-copy Buffers might still live DataMgr and hold ResultSets alive. Here
  // we clear these chunk to avoid unnecessary cached data. Remove in the reverse
  // order because later results can re-use and pin earlier ones. Some buffers
  // may still remain pinned by the final query result though.
  std::set<int, std::greater<int>> tmp_table_ids;
  for (auto& pr : temporary_tables_) {
    tmp_table_ids.insert(pr.second->tableId());
  }
  auto data_mgr = executor_->getDataMgr();
  ChunkKey prefix = {hdk::ResultSetRegistry::DB_ID, 0};
  for (auto table_id : tmp_table_ids) {
    prefix[1] = table_id;
    data_mgr->deleteChunksWithPrefix(prefix, MemoryLevel::CPU_LEVEL);
  }
}

ExecutionResult RelAlgExecutor::executeRelAlgQuery(const CompilationOptions& co,
                                                   const ExecutionOptions& eo,
                                                   const bool just_explain_plan) {
  CHECK(query_dag_);
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executeRelAlgQuery);

  auto run_query = [&](const CompilationOptions& co_in) {
    auto execution_result = executeRelAlgQueryNoRetry(co_in, eo, just_explain_plan);

    constexpr bool vlog_result_set_summary{false};
    if constexpr (vlog_result_set_summary) {
      for (size_t i = 0; i < execution_result.getToken()->resultSetCount(); ++i) {
        VLOG(1) << execution_result.getToken()->resultSet(i)->summaryToString();
      }
    }

    if (post_execution_callback_) {
      VLOG(1) << "Running post execution callback.";
      (*post_execution_callback_)();
    }
    return execution_result;
  };

  try {
    return run_query(co);
  } catch (const QueryMustRunOnCpu&) {
    if (!config_.exec.heterogeneous.allow_cpu_retry) {
      throw;
    }
  }
  LOG(INFO) << "Query unable to run in GPU mode, retrying on CPU";
  auto co_cpu = CompilationOptions::makeCpuOnly(co);

  return run_query(co_cpu);
}

std::string treeToString(const hdk::ir::Node* node,
                         bool stop_on_executed = true,
                         std::string prefix = "|") {
  std::stringstream ss;
  ss << prefix << node->toString() << std::endl;
  if (!stop_on_executed || !node->getResult()) {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      ss << treeToString(node->getInput(i), stop_on_executed, prefix + "----");
    }
  }
  return ss.str();
}

void printTree(const hdk::ir::Node* node,
               bool stop_on_executed = true,
               std::string prefix = "|") {
  std::cout << treeToString(node, stop_on_executed, prefix);
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryNoRetry(const CompilationOptions& co,
                                                          const ExecutionOptions& eo,
                                                          const bool just_explain_plan) {
  INJECT_TIMER(executeRelAlgQueryNoRetry);
  auto timer = DEBUG_TIMER(__func__);
  auto timer_setup = DEBUG_TIMER("Query pre-execution steps");

  query_dag_->resetQueryExecutionState();
  const auto ra = query_dag_->getRootNode();

  // capture the lock acquistion time
  auto clock_begin = timer_start();
  if (config_.exec.watchdog.enable_dynamic) {
    executor_->resetInterrupt();
  }

  int64_t queue_time_ms = timer_stop(clock_begin);
  ScopeGuard row_set_holder = [this] { cleanupPostExecution(); };
  const auto col_descs = get_physical_inputs(ra);
  const auto phys_table_ids = get_physical_table_inputs(ra);
  executor_->setSchemaProvider(schema_provider_);
  executor_->setupCaching(data_provider_, col_descs, phys_table_ids);

  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  hdk::QueryExecutionSequence query_seq(ra, executor_->getConfigPtr());
  if (just_explain_plan) {
    std::stringstream ss;
    std::vector<const hdk::ir::Node*> nodes;
    for (size_t i = 0; i < query_seq.size(); i++) {
      nodes.emplace_back(query_seq.step(i));
    }
    size_t ctr = nodes.size();
    size_t tab_ctr = 0;
    for (auto& body : boost::adaptors::reverse(nodes)) {
      const auto index = ctr--;
      const auto tabs = std::string(tab_ctr++, '\t');
      CHECK(body);
      ss << tabs << std::to_string(index) << " : " << body->toString() << "\n";
      if (auto sort = dynamic_cast<const hdk::ir::Sort*>(body)) {
        ss << tabs << "  : " << sort->getInput(0)->toString() << "\n";
      }
    }
    const auto& subqueries = getSubqueries();
    if (!subqueries.empty()) {
      ss << "Subqueries: "
         << "\n";
      for (const auto& subquery : subqueries) {
        const auto ra = subquery->node();
        ss << "\t" << ra->toString() << "\n";
      }
    }
    auto rs = std::make_shared<ResultSet>(ss.str());
    return registerResultSetTable({rs}, {}, true);
  }

  if (eo.find_push_down_candidates) {
    // this extra logic is mainly due to current limitations on multi-step queries
    // and/or subqueries.
    return executeRelAlgQueryWithFilterPushDown(query_seq, co, eo, queue_time_ms);
  }
  timer_setup.stop();

  // Dispatch the subqueries first
  for (auto& subquery : getSubqueries()) {
    auto subquery_ra = subquery->node();
    CHECK(subquery_ra);
    if (subquery_ra->hasContextData()) {
      continue;
    }

    RelAlgExecutor ra_executor(executor_, schema_provider_);
    hdk::QueryExecutionSequence subquery_seq(subquery_ra, executor_->getConfigPtr());
    ra_executor.execute(subquery_seq, co, eo, 0);
  }

  auto shared_res = execute(query_seq, co, eo, queue_time_ms);
  return std::move(*shared_res);
}

AggregatedColRange RelAlgExecutor::computeColRangesCache() {
  AggregatedColRange agg_col_range_cache;
  const auto col_descs = get_physical_inputs(getRootNode());
  return executor_->computeColRangesCache(col_descs);
}

StringDictionaryGenerations RelAlgExecutor::computeStringDictionaryGenerations() {
  const auto col_descs = get_physical_inputs(getRootNode());
  return executor_->computeStringDictionaryGenerations(col_descs);
}

TableGenerations RelAlgExecutor::computeTableGenerations() {
  const auto phys_table_ids = get_physical_table_inputs(getRootNode());
  return executor_->computeTableGenerations(phys_table_ids);
}

Executor* RelAlgExecutor::getExecutor() const {
  return executor_;
}

void RelAlgExecutor::cleanupPostExecution() {
  CHECK(executor_);
  executor_->row_set_mem_owner_ = nullptr;
}

std::pair<std::vector<unsigned>, std::unordered_map<unsigned, JoinQualsPerNestingLevel>>
RelAlgExecutor::getJoinInfo(const hdk::ir::Node* root_node) {
  auto sort_node = dynamic_cast<const hdk::ir::Sort*>(root_node);
  if (sort_node) {
    // we assume that test query that needs join info does not contain any sort node
    return {};
  }
  auto work_unit = createWorkUnit(root_node, {}, ExecutionOptions::fromConfig(Config()));
  return {{}, getLeftDeepJoinTreesInfo()};
}

namespace {

inline void check_sort_node_source_constraint(const hdk::ir::Sort* sort) {
  CHECK_EQ(size_t(1), sort->inputCount());
  const auto source = sort->getInput(0);
  if (dynamic_cast<const hdk::ir::Sort*>(source)) {
    throw std::runtime_error("Sort node not supported as input to another sort");
  }
}

/**
 * Check if multifrag result of input would be OK for node execution.
 */
inline bool supportsMultifragInput(const hdk::ir::Node* node,
                                   const hdk::ir::Node* input) {
  if (node->is<hdk::ir::Sort>()) {
    node = node->getInput(0);
  }
  if (!node->hasInput(input)) {
    return true;
  }
  if (node->is<hdk::ir::Project>() &&
      node->as<hdk::ir::Project>()->hasWindowFunctionExpr()) {
    return false;
  }
  if (node->is<hdk::ir::LogicalUnion>()) {
    return false;
  }
  return true;
}

}  // namespace

void RelAlgExecutor::prepareLeafExecution(
    const AggregatedColRange& agg_col_range,
    const StringDictionaryGenerations& string_dictionary_generations,
    const TableGenerations& table_generations) {
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  if (config_.exec.watchdog.enable_dynamic) {
    executor_->resetInterrupt();
  }
  queue_time_ms_ = timer_stop(clock_begin);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>(
      data_provider_, Executor::getArenaBlockSize(), cpu_threads());
  executor_->string_dictionary_generations_ = string_dictionary_generations;
  executor_->table_generations_ = table_generations;
  executor_->agg_col_range_cache_ = agg_col_range;
}

std::shared_ptr<const ExecutionResult> RelAlgExecutor::execute(
    const hdk::QueryExecutionSequence& seq,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const int64_t queue_time_ms,
    const bool with_existing_temp_tables) {
  INJECT_TIMER(execute);
  auto timer = DEBUG_TIMER(__func__);
  if (!with_existing_temp_tables) {
    decltype(temporary_tables_)().swap(temporary_tables_);
  }
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  decltype(left_deep_join_info_)().swap(left_deep_join_info_);
  executor_->setSchemaProvider(schema_provider_);
  executor_->temporary_tables_ = &temporary_tables_;

  time(&now_);
  CHECK(seq.size());

  auto get_descriptor_count = [&seq, &eo]() -> size_t {
    if (eo.just_explain) {
      if (dynamic_cast<const hdk::ir::LogicalValues*>(seq.step(0))) {
        // run the logical values descriptor to generate the result set, then the next
        // descriptor to generate the explain
        CHECK_GE(seq.size(), size_t(2));
        return 2;
      } else {
        return 1;
      }
    } else {
      return seq.size();
    }
  };

  const auto exec_desc_count = get_descriptor_count();
  // this join info needs to be maintained throughout an entire query runtime
  for (size_t i = 0; i < exec_desc_count; i++) {
    VLOG(1) << "Executing query step " << i;
    // When we execute the last step, we expect the result to consist of a single
    // ResultSet unless said otherwise by the config. Also, check if following steps
    // can consume multifrag input.
    auto multifrag_result =
        eo.multifrag_result &&
        (i != (exec_desc_count - 1) || config_.exec.enable_multifrag_execution_result);
    if (multifrag_result) {
      for (size_t j = i + 1; j < exec_desc_count; j++) {
        if (!supportsMultifragInput(seq.step(j), seq.step(i))) {
          multifrag_result = false;
          break;
        }
      }
    }
    auto fixed_eo = eo.with_multifrag_result(multifrag_result);
    try {
      executeStep(seq.step(i), co, fixed_eo, queue_time_ms);
    } catch (const QueryMustRunOnCpu&) {
      CHECK(config_.exec.heterogeneous.enable_heterogeneous_execution ||
            co.device_type == ExecutorDeviceType::GPU);
      if (!config_.exec.heterogeneous.allow_query_step_cpu_retry) {
        throw;
      }
      LOG(INFO) << "Retrying current query step " << i << " on CPU";
      const auto co_cpu = CompilationOptions::makeCpuOnly(co);
      executeStep(seq.step(i), co_cpu, fixed_eo, queue_time_ms);
    } catch (const NativeExecutionError&) {
      if (!config_.exec.enable_interop) {
        throw;
      }
      auto eo_extern = eo;
      eo_extern.executor_type = ::ExecutorType::Extern;
      executeStep(seq.step(i), co, eo_extern, queue_time_ms);
    } catch (const RequestPartitionedAggregation& e) {
      executeStepWithPartitionedAggregation(
          seq.step(i), co, eo, e.estimatedBufferSize(), queue_time_ms);
    }
  }

  return seq.step(exec_desc_count - 1)->getResult();
}

void RelAlgExecutor::handleNop(RaExecutionDesc& ed) {
  // just set the result of the previous node as the result of no op
  auto body = ed.getBody();
  CHECK(dynamic_cast<const hdk::ir::Aggregate*>(body));
  CHECK_EQ(size_t(1), body->inputCount());
  const auto input = body->getInput(0);
  body->setOutputMetainfo(input->getOutputMetainfo());
  const auto it = temporary_tables_.find(-input->getId());
  CHECK(it != temporary_tables_.end());

  ed.setResult({it->second, input->getOutputMetainfo()});

  // set up temp table as it could be used by the outer query or next step
  addTemporaryTable(-body->getId(), it->second);
}

namespace {

const hdk::ir::Node* get_data_sink(const hdk::ir::Node* ra_node) {
  if (auto join = dynamic_cast<const hdk::ir::Join*>(ra_node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    return join;
  }
  if (!dynamic_cast<const hdk::ir::LogicalUnion*>(ra_node)) {
    CHECK_EQ(size_t(1), ra_node->inputCount());
  }
  auto only_src = ra_node->getInput(0);
  const bool is_join = dynamic_cast<const hdk::ir::Join*>(only_src);
  return is_join ? only_src : ra_node;
}

std::unordered_map<const hdk::ir::Node*, int> get_input_nest_levels(
    const hdk::ir::Node* ra_node,
    const std::vector<size_t>& input_permutation) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const hdk::ir::Node*, int> input_to_nest_level;
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_node_idx =
        input_permutation.empty() ? input_idx : input_permutation[input_idx];
    const auto input_ra = data_sink_node->getInput(input_node_idx);
    // Having a non-zero mapped value (input_idx) results in the query being
    // interpretted as a JOIN within CodeGenerator::codegenColVar() due to rte_idx
    // being set to the mapped value (input_idx) which originates here. This would be
    // incorrect for UNION.
    size_t const idx =
        dynamic_cast<const hdk::ir::LogicalUnion*>(ra_node) ? 0 : input_idx;
    const auto it_ok = input_to_nest_level.emplace(input_ra, idx);
    CHECK(it_ok.second);
    LOG_IF(INFO, !input_permutation.empty())
        << "Assigned input " << input_ra->toString() << " to nest level " << input_idx;
  }
  return input_to_nest_level;
}

hdk::ir::ExprPtr set_transient_dict_maybe(hdk::ir::ExprPtr expr) {
  try {
    return set_transient_dict(fold_expr(expr.get()));
  } catch (const OverflowOrUnderflow& e) {
    throw e;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught exception trying to set transient dictionary source: "
                 << e.what();
    return expr;
  }
}

hdk::ir::ExprPtr cast_dict_to_none(const hdk::ir::ExprPtr& input) {
  auto input_type = input->type();
  if (input_type->isExtDictionary()) {
    return input->cast(input_type->ctx().text(input_type->nullable()));
  }
  return input;
}

bool is_count_distinct(const hdk::ir::Expr* expr) {
  const auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(expr);
  return agg_expr && agg_expr->isDistinct();
}

bool is_agg(const hdk::ir::Expr* expr) {
  const auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(expr);
  if (agg_expr && agg_expr->containsAgg()) {
    auto agg_type = agg_expr->aggType();
    if (agg_type == hdk::ir::AggType::kMin || agg_type == hdk::ir::AggType::kMax ||
        agg_type == hdk::ir::AggType::kSum || agg_type == hdk::ir::AggType::kAvg) {
      return true;
    }
  }
  return false;
}

const hdk::ir::Type* canonicalTypeForExpr(const hdk::ir::Expr& expr) {
  if (is_count_distinct(&expr)) {
    return expr.type()->ctx().int64();
  }
  auto res = expr.type()->canonicalize();
  if (is_agg(&expr)) {
    res = res->withNullable(true);
  }
  return res;
}

std::vector<hdk::ir::TargetMetaInfo> get_targets_meta(
    const hdk::ir::Node* node,
    const std::vector<const hdk::ir::Expr*>& target_exprs) {
  std::vector<hdk::ir::TargetMetaInfo> targets_meta;
  CHECK_EQ(node->size(), target_exprs.size());
  for (size_t i = 0; i < node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(node->getFieldName(i),
                              canonicalTypeForExpr(*target_exprs[i]));
  }
  return targets_meta;
}

bool is_agg_step(const hdk::ir::Node* node) {
  if (node->getResult() || node->is<hdk::ir::Scan>()) {
    return false;
  }
  if (node->is<hdk::ir::Aggregate>()) {
    return true;
  }
  if (auto shuffle = node->as<hdk::ir::Shuffle>()) {
    return shuffle->isCount();
  }
  for (size_t i = 0; i < node->inputCount(); ++i) {
    if (is_agg_step(node->getInput(i))) {
      return true;
    }
  }
  return false;
}

class NonGroupbyInputColIndicesCollector
    : public hdk::ir::ExprCollector<std::set<unsigned>,
                                    NonGroupbyInputColIndicesCollector> {
 public:
  NonGroupbyInputColIndicesCollector(const hdk::ir::Aggregate* agg) : agg_(agg){};

  static std::set<unsigned> collect(const hdk::ir::Aggregate* agg) {
    return BaseClass::collect(agg->getAggs(), agg);
  }

 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (col_ref->index() >= agg_->getGroupByCount()) {
      result_.insert(col_ref->index());
    }
  }

  const hdk::ir::Aggregate* agg_;
};

void buildUsedAggColumns(const hdk::ir::Aggregate* agg,
                         hdk::ir::ExprPtrVector& out_exprs,
                         std::vector<std::string>& out_fields,
                         std::unordered_map<unsigned, unsigned>& out_mapping) {
  auto input = agg->getInput(0);
  for (size_t i = 0; i < agg->getGroupByCount(); ++i) {
    out_exprs.push_back(hdk::ir::getNodeColumnRef(input, static_cast<unsigned>(i)));
    out_fields.push_back(input->getFieldName(i));
  }
  auto data_cols = NonGroupbyInputColIndicesCollector::collect(agg);
  for (auto& idx : data_cols) {
    out_mapping[idx] = static_cast<unsigned>(out_exprs.size());
    out_exprs.push_back(hdk::ir::getNodeColumnRef(input, idx));
    out_fields.push_back(input->getFieldName(idx));
  }
}

// Check of node should be executed before shuffle can be applied to it.
// Shuffle goes in two passes and, due to lack of appropriate cost model,
// we prefer shuffle's input to be materialized. Exception is when it is
// a simple projection.
// TODO: enable more cases when cheap operation can be merged into shuffle.
bool shouldMaterializeShuffleInput(const hdk::ir::Node* node) {
  if (node->getResult() || node->is<hdk::ir::Scan>()) {
    return false;
  }

  auto proj = node->as<hdk::ir::Project>();
  if (!proj) {
    return true;
  }

  return !proj->isSimple() || shouldMaterializeShuffleInput(proj->getInput(0));
}

}  // namespace

hdk::ir::ExprPtr set_transient_dict(const hdk::ir::ExprPtr expr) {
  auto type = expr->type();
  if (!type->isString()) {
    return expr;
  }
  auto transient_dict_type = type->ctx().extDict(type, TRANSIENT_DICT_ID);
  return expr->cast(transient_dict_type);
}

hdk::ir::ExprPtr translate(const hdk::ir::Expr* expr,
                           const RelAlgTranslator& translator,
                           ::ExecutorType executor_type) {
  auto res = translator.normalize(expr);
  res = rewrite_array_elements(res.get());
  res = rewrite_expr(res.get());
  if (executor_type == ExecutorType::Native) {
    // This is actually added to get full match of translated legacy
    // rex expressions and new Exprs. It's done only for testing purposes
    // and shouldn't have any effect on functionality and performance.
    // TODO: remove when rex are not used anymore
    if (auto* agg = dynamic_cast<const hdk::ir::AggExpr*>(res.get())) {
      if (agg->arg()) {
        auto new_arg = set_transient_dict_maybe(agg->argShared());
        if (new_arg != agg->argShared()) {
          res = hdk::ir::makeExpr<hdk::ir::AggExpr>(agg->type(),
                                                    agg->aggType(),
                                                    new_arg,
                                                    agg->isDistinct(),
                                                    agg->arg1Shared(),
                                                    agg->interpolation());
        }
      }
    } else {
      res = set_transient_dict_maybe(res);
    }
  } else {
    res = cast_dict_to_none(fold_expr(res.get()));
  }
  return res;
}

void RelAlgExecutor::executeStepWithPartitionedAggregation(const hdk::ir::Node* step_root,
                                                           const CompilationOptions& co,
                                                           const ExecutionOptions& eo,
                                                           size_t estimated_buffer_size,
                                                           const int64_t queue_time_ms) {
  auto sort = step_root->as<hdk::ir::Sort>();
  auto agg = sort ? sort->getInput(0)->as<hdk::ir::Aggregate>()
                  : step_root->as<hdk::ir::Aggregate>();
  CHECK(agg);

  // Materialize data to shuffle if required.
  auto agg_input = agg->getInput(0);
  auto agg_input_shared = const_cast<hdk::ir::Aggregate*>(agg)->getAndOwnInput(0);
  std::shared_ptr<hdk::ir::Project> proj;
  std::unordered_map<unsigned, unsigned> input_map;
  if (shouldMaterializeShuffleInput(agg_input)) {
    // Make additional projection to possibly filter out unused columns (e.g. in
    // case of join).
    hdk::ir::ExprPtrVector exprs;
    std::vector<std::string> fields;
    buildUsedAggColumns(agg, exprs, fields, input_map);
    proj = std::make_shared<hdk::ir::Project>(
        std::move(exprs), std::move(fields), agg_input_shared);
    VLOG(1) << "Materializing data for partitioning.";
    executeStep(proj.get(), co, eo, queue_time_ms);
  }

  // Now that data to shuffle is materialized, we can start shuffling. As the first
  // step, we compute the desired number of resulting partitions and compute their sizes.
  // By default, we want to have enough partitions to load all cores but not to slow
  // down partitioning too much by too many output streams.
  size_t min_partitions = config_.exec.group_by.min_partitions
                              ? config_.exec.group_by.min_partitions
                              : cpu_threads() * 2;
  size_t max_partitions = std::max(config_.exec.group_by.max_partitions, min_partitions);
  // For now, we require number of partitions to be power of 2 to avoid division
  // operation in partitioning code.
  min_partitions = shared::roundUpToPow2(min_partitions);
  max_partitions = shared::roundUpToPow2(max_partitions);
  size_t partitions = min_partitions;
  // Increase number of partitions until we achieve target output buffer size or
  // hit the partitioning limit.
  while (partitions < max_partitions &&
         estimated_buffer_size / partitions >
             config_.exec.group_by.partitioning_buffer_target_size) {
    partitions = partitions * 2;
  }
  VLOG(1) << "Selected to use " << partitions << " partitions. Used range is ["
          << min_partitions << ", " << max_partitions << "]. Estimated buffer size is "
          << estimated_buffer_size;
  hdk::ir::ShuffleFunction shuffle_fn{hdk::ir::ShuffleFunction::kHash, partitions};
  VLOG(1) << "Selected shuffle function is " << shuffle_fn;
  hdk::ir::NodePtr shuffle_input = proj ? proj : agg_input_shared;
  hdk::ir::ExprPtrVector shuffle_keys;
  for (unsigned i = 0; i < static_cast<unsigned>(agg->getGroupByCount()); ++i) {
    shuffle_keys.push_back(getNodeColumnRef(shuffle_input.get(), i));
  }
  hdk::ir::NodePtr count_shuffle_node = std::make_shared<hdk::ir::Shuffle>(
      shuffle_keys,
      hdk::ir::makeExpr<hdk::ir::AggExpr>(hdk::ir::Context::defaultCtx().int64(false),
                                          hdk::ir::AggType::kCount,
                                          nullptr,
                                          false,
                                          nullptr),
      "part_size",
      shuffle_fn,
      shuffle_input);
  VLOG(1) << "Execute COUNT(*) for data shuffle.";
  {
    auto timer = DEBUG_TIMER("Partitions histogram computation");
    executeStep(
        count_shuffle_node.get(), co, eo.with_columnar_output(true), queue_time_ms);
  }

  // With partition sizes now known, start actual data shuffling.
  hdk::ir::ExprPtrVector shuffle_input_cols;
  std::vector<std::string> fields;
  if (proj) {
    shuffle_input_cols = hdk::ir::getNodeColumnRefs(proj.get());
    fields = proj->getFields();
  } else {
    hdk::ir::ExprPtrVector exprs;
    buildUsedAggColumns(agg, shuffle_input_cols, fields, input_map);
  }
  hdk::ir::ExprPtrVector shuffle_exprs;
  auto offsets_ref = hdk::ir::getNodeColumnRef(count_shuffle_node.get(), 0);
  hdk::ir::NodePtr shuffle_node = std::make_shared<hdk::ir::Shuffle>(
      std::move(shuffle_keys),
      std::move(shuffle_input_cols),
      std::move(fields),
      shuffle_fn,
      std::vector<hdk::ir::NodePtr>({shuffle_input, count_shuffle_node}));
  VLOG(1) << "Execute data shuffle.";
  auto shuffle_eo = eo;
  shuffle_eo.output_columnar_hint = true;
  shuffle_eo.preserve_order = false;
  auto shuffle_co = co;
  shuffle_co.allow_lazy_fetch = false;
  {
    auto timer = DEBUG_TIMER("Data shuffling");
    executeStep(shuffle_node.get(), shuffle_co, shuffle_eo, queue_time_ms);
  }

  // Now we can remove projection and its result to free memory.
  if (proj) {
    temporary_tables_.erase(-proj->getId());
    proj.reset();
  }

  // Currently, we merge shuffle node with simple projections only. Therefore, we can
  // assign original table stats to shuffling results to avoid metadata computation.
  maybeCopyTableStatsFromInput(shuffle_node.get());

  // Create new aggregation node and execute it.
  auto part_agg = std::make_shared<hdk::ir::Aggregate>(
      agg->getGroupByCount(), agg->getAggs(), agg->getFields(), agg_input_shared);
  part_agg->replaceInput(agg_input_shared, shuffle_node, input_map);
  part_agg->setPartitioned(true);
  // Use max partition size for a new entry count guess.
  auto count_res_token = temporary_tables_.at(-count_shuffle_node->getId());
  const uint64_t* size_buf = reinterpret_cast<const uint64_t*>(
      count_res_token->resultSet(count_res_token->resultSetCount() - 1)
          ->getStorage()
          ->getUnderlyingBuffer());
  auto max_partition_size = *std::max_element(size_buf, size_buf + partitions);
  part_agg->setBufferEntryCountHint(max_partition_size * 2);
  VLOG(1) << "Using buffer entry count hint for partitioned aggregation: "
          << part_agg->bufferEntryCountHint();
  hdk::ir::NodePtr new_root = part_agg;
  if (sort) {
    new_root = sort->deepCopy();
    new_root->replaceInput(new_root->getAndOwnInput(0), part_agg);
  }
  VLOG(1) << "Execute partitioned aggregation.";
  {
    auto timer = DEBUG_TIMER("Partitioned aggregation");
    executeStep(new_root.get(), co, eo, queue_time_ms);
  }

  // Register result as a temporary table for the original node.
  addTemporaryTable(-step_root->getId(), new_root->getResult()->getToken());
  step_root->setResult(new_root->getResult());
  // Remove temporary tables we don't need anymore.
  temporary_tables_.erase(-count_shuffle_node->getId());
  temporary_tables_.erase(-shuffle_node->getId());
  temporary_tables_.erase(-part_agg->getId());
  temporary_tables_.erase(-new_root->getId());
}

void RelAlgExecutor::maybeCopyTableStatsFromInput(const hdk::ir::Node* node) {
  std::vector<int> col_mapping;
  col_mapping.reserve(node->size());
  // Stats copy is supported for shuffle and simple projections only.
  if (node->is<hdk::ir::Project>()) {
    auto proj = node->as<hdk::ir::Project>();
    if (!proj->isSimple()) {
      VLOG(1) << "Cannot copy table stats for non-simple projection.";
      return;
    }
    for (auto& expr : proj->getExprs()) {
      col_mapping.push_back(expr->as<hdk::ir::ColumnRef>()->index());
    }
  } else if (node->is<hdk::ir::Shuffle>()) {
    for (auto& expr : node->as<hdk::ir::Shuffle>()->exprs()) {
      CHECK(expr->is<hdk::ir::ColumnRef>());
      col_mapping.push_back(expr->as<hdk::ir::ColumnRef>()->index());
    }
  } else {
    VLOG(1) << "Cannot copy table stats for node " << node->toString();
    return;
  }

  // We can traverse through a chain of simple projections to the original data source.
  auto data_source = node->getInput(0);
  while (!data_source->getResult() && !data_source->is<hdk::ir::Scan>()) {
    auto proj = data_source->as<hdk::ir::Project>();
    if (!proj || !proj->isSimple()) {
      VLOG(1) << "Cannot copy table stats due to non-simple projection. "
              << node->toString();
      return;
    }
    for (size_t i = 0; i < col_mapping.size(); ++i) {
      auto idx = static_cast<size_t>(col_mapping[i]);
      CHECK_LT(idx, proj->size());
      col_mapping[i] = proj->getExpr(idx)->as<hdk::ir::ColumnRef>()->index();
    }
    data_source = data_source->getInput(0);
  }

  auto input_token =
      data_source->getResult() ? data_source->getResult()->getToken().get() : nullptr;
  auto input_scan = data_source->getResult() ? nullptr : data_source->as<hdk::ir::Scan>();
  int input_db_id = input_token ? input_token->dbId() : input_scan->getDatabaseId();
  int input_table_id = input_token ? input_token->tableId() : input_scan->getTableId();
  auto input_meta = data_provider_->getTableMetadata(input_db_id, input_table_id);
  if (input_meta->hasComputedTableStats()) {
    auto& orig_stats = input_meta->getTableStats();
    auto target_token = node->getResult()->getToken();
    TableStats stats;
    for (size_t i = 0; i < col_mapping.size(); ++i) {
      auto target_col_id = target_token->columnId(i);
      auto input_col_id = input_token
                              ? input_token->columnId(col_mapping[i])
                              : input_scan->getColumnInfo(col_mapping[i])->column_id;
      CHECK(orig_stats.count(input_col_id))
          << "Cannot find stats for column " << input_col_id
          << ". data_source=" << data_source->toString();
      stats.emplace(target_col_id, orig_stats.at(input_col_id));
    }
    target_token->setTableStats(std::move(stats));
    VLOG(1) << "Copy table stats from " << input_db_id << ":" << input_table_id << " to "
            << target_token->dbId() << ":" << target_token->tableId();
  } else {
    VLOG(1) << "Cannot copy table stats because original table stats are unavailable.";
  }
}

void RelAlgExecutor::executeStep(const hdk::ir::Node* step_root,
                                 const CompilationOptions& co,
                                 const ExecutionOptions& eo,
                                 const int64_t queue_time_ms) {
  VLOG(1) << "Executing query step:\n" << treeToString(step_root, true);
  ExecutionResult res;
  try {
    // TODO: move allow_speculative_sort to ExecutionOptions?
    res = executeStep(step_root, co, eo, queue_time_ms, true);
  } catch (const SpeculativeTopNFailed& e) {
    VLOG(1) << "Retrying step with disabled speculative TopN.";
    res = executeStep(step_root, co, eo, queue_time_ms, false);
  }

  auto shared_res = std::make_shared<ExecutionResult>(std::move(res));
  step_root->setResult(shared_res);
  // Logical values are always executed and ignore just_explain flag.
  if (!eo.just_explain || step_root->is<hdk::ir::LogicalValues>()) {
    addTemporaryTable(-step_root->getId(), shared_res->getToken());
  }
}

ExecutionResult RelAlgExecutor::executeStep(const hdk::ir::Node* step_root,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            const int64_t queue_time_ms,
                                            bool allow_speculative_sort) {
  auto timer = DEBUG_TIMER(__func__);
  WindowProjectNodeContext::reset(executor_);
  // Currently, table functions and UNION ALL nodes are not merged
  // with other nodes. We don't use WorkUnitBuilder for them.
  if (auto logical_values = step_root->as<hdk::ir::LogicalValues>()) {
    return executeLogicalValues(logical_values, eo);
  }

  WorkUnit work_unit = createWorkUnit(step_root, co, eo, allow_speculative_sort);

  auto sort = step_root->as<hdk::ir::Sort>();
  ExecutionOptions eo_with_limit =
      eo.with_just_validate(eo.just_validate || (sort && sort->isEmptyResult()));
  // Use additional result fragments sort for UNION ALL case.
  // Detect it via a check for two outer tables in the input
  // descriptors vector.
  if (work_unit.exe_unit.input_descs.size() >= 2 &&
      work_unit.exe_unit.input_descs[0].getNestLevel() ==
          work_unit.exe_unit.input_descs[1].getNestLevel()) {
    eo_with_limit = eo_with_limit.with_preserve_order(true);
  }

  bool cpu_only = false;
  if (auto project = step_root->as<hdk::ir::Project>()) {
    if (project->isSimple()) {
      const auto input_node = project->getInput(0);
      if (input_node->is<hdk::ir::Sort>()) {
        cpu_only = true;
        // TODO: why do we need scan limit here?
        auto token = get_temporary_table(&temporary_tables_, -input_node->getId());
        work_unit.exe_unit.scan_limit = token->rowCount();
      }
    }
  }

  // Request single ResultSet in the resulting table if we are going to sort it.
  if (sort) {
    eo_with_limit.multifrag_result = false;
  }
  auto res = executeWorkUnit(work_unit,
                             step_root->getOutputMetainfo(),
                             is_agg_step(step_root),
                             cpu_only ? CompilationOptions::makeCpuOnly(co) : co,
                             eo_with_limit,
                             queue_time_ms);

  if (sort) {
    if (res.isFilterPushDownEnabled()) {
      return res;
    }
    auto rows_to_sort = res.getRows();
    if (eo.just_explain) {
      return res;
    }
    const size_t limit = sort->getLimit();
    const size_t offset = sort->getOffset();
    if (sort->collationCount() != 0 && !rows_to_sort->definitelyHasNoRows() &&
        !use_speculative_top_n(work_unit.exe_unit, rows_to_sort->getQueryMemDesc())) {
      const size_t top_n = limit == 0 ? 0 : limit + offset;
      sortResultSet(rows_to_sort.get(),
                    work_unit.exe_unit.sort_info.order_entries,
                    top_n,
                    executor_);
    }
    if (limit || offset) {
      rows_to_sort->dropFirstN(offset);
      if (limit) {
        rows_to_sort->keepFirstN(limit);
      }
    }
    // Already registered table cannot be used after sort, limit, and offset are applied.
    // Register a new table. The previous token in res should die at the exit and remove
    // the previous table from the registry.
    return registerResultSetTable({rows_to_sort}, res.getTargetsMeta(), eo.just_explain);
  }

  return res;
}

namespace {

// Returns true iff the execution unit contains window functions.
bool is_window_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return std::any_of(ra_exe_unit.target_exprs.begin(),
                     ra_exe_unit.target_exprs.end(),
                     [](const hdk::ir::Expr* expr) {
                       return dynamic_cast<const hdk::ir::WindowFunction*>(expr);
                     });
}

// Creates a new expression which has the range table index set to 1. This is needed
// to reuse the hash join construction helpers to generate a hash table for the window
// function partition: create an equals expression with left and right sides identical
// except for the range table index.
hdk::ir::ExprPtr transform_to_inner(const hdk::ir::Expr* expr) {
  const auto tuple = dynamic_cast<const hdk::ir::ExpressionTuple*>(expr);
  if (tuple) {
    std::vector<hdk::ir::ExprPtr> transformed_tuple;
    for (const auto& element : tuple->tuple()) {
      transformed_tuple.push_back(transform_to_inner(element.get()));
    }
    return hdk::ir::makeExpr<hdk::ir::ExpressionTuple>(transformed_tuple);
  }
  const auto col = dynamic_cast<const hdk::ir::ColumnVar*>(expr);
  if (!col) {
    throw std::runtime_error("Only columns supported in the window partition for now");
  }
  return hdk::ir::makeExpr<hdk::ir::ColumnVar>(col->columnInfo(), 1);
}

}  // namespace

void RelAlgExecutor::computeWindow(const RelAlgExecutionUnit& ra_exe_unit,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   ColumnCacheMap& column_cache_map,
                                   const int64_t queue_time_ms) {
  auto query_infos = get_table_infos(ra_exe_unit.input_descs, executor_);
  CHECK_EQ(query_infos.size(), size_t(1));
  if (query_infos.front().info->fragments.size() != 1) {
    throw std::runtime_error(
        "Only single fragment tables supported for window functions for now");
  }
  if (eo.executor_type == ::ExecutorType::Extern) {
    return;
  }
  query_infos.push_back(query_infos.front());
  auto window_project_node_context = WindowProjectNodeContext::create(executor_);
  for (size_t target_index = 0; target_index < ra_exe_unit.target_exprs.size();
       ++target_index) {
    const auto& target_expr = ra_exe_unit.target_exprs[target_index];
    const auto window_func = dynamic_cast<const hdk::ir::WindowFunction*>(target_expr);
    if (!window_func) {
      continue;
    }
    // Always use baseline layout hash tables for now, make the expression a tuple.
    const auto& partition_keys = window_func->partitionKeys();
    std::shared_ptr<const hdk::ir::BinOper> partition_key_cond;
    if (partition_keys.size() >= 1) {
      hdk::ir::ExprPtr partition_key_tuple;
      if (partition_keys.size() > 1) {
        partition_key_tuple = hdk::ir::makeExpr<hdk::ir::ExpressionTuple>(partition_keys);
      } else {
        CHECK_EQ(partition_keys.size(), size_t(1));
        partition_key_tuple = partition_keys.front();
      }
      // Creates a tautology equality with the partition expression on both sides.
      partition_key_cond = hdk::ir::makeExpr<hdk::ir::BinOper>(
          target_expr->ctx().boolean(),
          hdk::ir::OpType::kBwEq,
          hdk::ir::Qualifier::kOne,
          partition_key_tuple,
          transform_to_inner(partition_key_tuple.get()));
    }
    auto context =
        createWindowFunctionContext(window_func,
                                    partition_key_cond /*nullptr if no partition key*/,
                                    ra_exe_unit,
                                    query_infos,
                                    co,
                                    column_cache_map,
                                    executor_->getRowSetMemoryOwner());
    context->compute();
    window_project_node_context->addWindowFunctionContext(std::move(context),
                                                          target_index);
  }
}

std::unique_ptr<WindowFunctionContext> RelAlgExecutor::createWindowFunctionContext(
    const hdk::ir::WindowFunction* window_func,
    const std::shared_ptr<const hdk::ir::BinOper>& partition_key_cond,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const CompilationOptions& co,
    ColumnCacheMap& column_cache_map,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  const size_t elem_count = query_infos.front().info->fragments.front().getNumTuples();
  const auto memory_level = co.device_type == ExecutorDeviceType::GPU
                                ? MemoryLevel::GPU_LEVEL
                                : MemoryLevel::CPU_LEVEL;
  std::unique_ptr<WindowFunctionContext> context;
  if (partition_key_cond) {
    const auto join_table_or_err =
        executor_->buildHashTableForQualifier(partition_key_cond,
                                              query_infos,
                                              memory_level,
                                              JoinType::INVALID,  // for window function
                                              HashType::OneToMany,
                                              data_provider_,
                                              column_cache_map,
                                              ra_exe_unit.hash_table_build_plan_dag,
                                              ra_exe_unit.table_id_to_node_map);
    if (!join_table_or_err.fail_reason.empty()) {
      throw std::runtime_error(join_table_or_err.fail_reason);
    }
    CHECK(join_table_or_err.hash_table->getHashType() == HashType::OneToMany);
    context = std::make_unique<WindowFunctionContext>(window_func,
                                                      config_,
                                                      join_table_or_err.hash_table,
                                                      elem_count,
                                                      co.device_type,
                                                      row_set_mem_owner);
  } else {
    context = std::make_unique<WindowFunctionContext>(
        window_func, config_, elem_count, co.device_type, row_set_mem_owner);
  }
  const auto& order_keys = window_func->orderKeys();
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  for (const auto& order_key : order_keys) {
    const auto order_col = std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(order_key);
    if (!order_col) {
      throw std::runtime_error("Only order by columns supported for now");
    }
    const int8_t* column;
    size_t join_col_elem_count;
    std::tie(column, join_col_elem_count) =
        ColumnFetcher::getOneColumnFragment(executor_,
                                            *order_col,
                                            query_infos.front().info->fragments.front(),
                                            memory_level,
                                            0,
                                            nullptr,
                                            chunks_owner,
                                            data_provider_,
                                            column_cache_map);

    CHECK_EQ(join_col_elem_count, elem_count);
    context->addOrderColumn(column, order_col.get(), chunks_owner);
  }
  return context;
}

ExecutionResult RelAlgExecutor::executeLogicalValues(
    const hdk::ir::LogicalValues* logical_values,
    const ExecutionOptions& eo) {
  auto timer = DEBUG_TIMER(__func__);
  QueryMemoryDescriptor query_mem_desc(executor_->getDataMgr(),
                                       executor_->getConfigPtr(),
                                       logical_values->getNumRows(),
                                       QueryDescriptionType::Projection,
                                       /*is_table_function=*/false);

  auto tuple_type = logical_values->getTupleType();
  for (size_t i = 0; i < tuple_type.size(); ++i) {
    auto& target_meta_info = tuple_type[i];
    if (target_meta_info.type()->isString() || target_meta_info.type()->isArray()) {
      throw std::runtime_error("Variable length types not supported in VALUES yet.");
    }
    if (target_meta_info.type()->isNull()) {
      // replace w/ bigint
      tuple_type[i] = hdk::ir::TargetMetaInfo(target_meta_info.get_resname(),
                                              hdk::ir::Context::defaultCtx().int64());
    }
    query_mem_desc.addColSlotInfo({std::make_tuple(tuple_type[i].type()->size(), 8)});
  }
  logical_values->setOutputMetainfo(tuple_type);

  std::vector<TargetInfo> target_infos;
  for (const auto& tuple_type_component : tuple_type) {
    target_infos.emplace_back(TargetInfo{false,
                                         hdk::ir::AggType::kCount,
                                         tuple_type_component.type(),
                                         nullptr,
                                         false,
                                         false});
  }

  std::shared_ptr<ResultSet> rs{
      ResultSetLogicalValuesBuilder{logical_values,
                                    target_infos,
                                    ExecutorDeviceType::CPU,
                                    query_mem_desc,
                                    executor_->getRowSetMemoryOwner(),
                                    executor_}
          .build()};

  // Ignore just_explain flag for logical values node which is always executed
  // and whose result can be used for actual explain description generation.
  return registerResultSetTable({rs}, tuple_type, false);
}

namespace {

/**
 *  Upper bound estimation for the number of groups. Not strictly correct and not
 * tight, but if the tables involved are really small we shouldn't waste time doing
 * the NDV estimation. We don't account for cross-joins and / or group by unnested
 * array, which is the reason this estimation isn't entirely reliable.
 */
size_t groups_approx_upper_bound(const std::vector<InputTableInfo>& table_infos) {
  CHECK(!table_infos.empty());
  const auto& first_table = table_infos.front();
  size_t max_num_groups = first_table.info->getNumTuplesUpperBound();
  for (const auto& table_info : table_infos) {
    if (table_info.info->getNumTuplesUpperBound() > max_num_groups) {
      max_num_groups = table_info.info->getNumTuplesUpperBound();
    }
  }
  return std::max(max_num_groups, size_t(1));
}

/**
 * Determines whether a query needs to compute the size of its output buffer. Returns
 * true for projection queries with no LIMIT or a LIMIT that exceeds the high scan
 * limit threshold (meaning it would be cheaper to compute the number of rows passing
 * or use the bump allocator than allocate the current scan limit per GPU)
 */
bool compute_output_buffer_size(const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const hdk::ir::AggExpr*>(target_expr)) {
      return false;
    }
  }
  if (ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit || ra_exe_unit.scan_limit > Executor::high_scan_limit)) {
    return true;
  }
  return false;
}

inline bool exe_unit_has_quals(const RelAlgExecutionUnit ra_exe_unit) {
  return !(ra_exe_unit.quals.empty() && ra_exe_unit.join_quals.empty() &&
           ra_exe_unit.simple_quals.empty());
}

RelAlgExecutionUnit decide_approx_count_distinct_implementation(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor,
    const ExecutorDeviceType device_type_in,
    std::vector<hdk::ir::ExprPtr>& target_exprs_owned) {
  RelAlgExecutionUnit ra_exe_unit = ra_exe_unit_in;
  for (size_t i = 0; i < ra_exe_unit.target_exprs.size(); ++i) {
    const auto target_expr = ra_exe_unit.target_exprs[i];
    const auto agg_info =
        get_target_info(target_expr, executor->getConfig().exec.group_by.bigint_count);
    if (agg_info.agg_kind != hdk::ir::AggType::kApproxCountDistinct) {
      continue;
    }
    auto agg_expr = target_expr->as<hdk::ir::AggExpr>();
    CHECK(agg_expr);
    const auto arg = agg_expr->argShared();
    CHECK(arg);
    auto arg_type = arg->type();
    // Avoid calling getExpressionRange for variable length types (string and array),
    // it'd trigger an assertion since that API expects to be called only for types
    // for which the notion of range is well-defined. A bit of a kludge, but the
    // logic to reject these types anyway is at lower levels in the stack and not
    // really worth pulling into a separate function for now.
    if (!(arg_type->isNumber() || arg_type->isBoolean() || arg_type->isDateTime() ||
          (arg_type->isExtDictionary()))) {
      continue;
    }
    const auto arg_range = getExpressionRange(arg.get(), table_infos, executor);
    if (arg_range.getType() != ExpressionRangeType::Integer) {
      continue;
    }
    const auto device_type = device_type_in;
    const auto bitmap_sz_bits = arg_range.getIntMax() - arg_range.getIntMin() + 1;
    const auto sub_bitmap_count =
        get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
    int64_t approx_bitmap_sz_bits{0};
    const auto error_rate =
        agg_expr->arg1() ? agg_expr->arg1()->as<hdk::ir::Constant>() : nullptr;
    if (error_rate) {
      CHECK(error_rate->type()->isInt32());
      CHECK_GE(error_rate->value().intval, 1);
      approx_bitmap_sz_bits = hll_size_for_rate(error_rate->value().intval);
    } else {
      approx_bitmap_sz_bits = executor->getConfig().exec.group_by.hll_precision_bits;
    }
    CountDistinctDescriptor approx_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                       arg_range.getIntMin(),
                                                       approx_bitmap_sz_bits,
                                                       true,
                                                       device_type,
                                                       sub_bitmap_count};
    CountDistinctDescriptor precise_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                        arg_range.getIntMin(),
                                                        bitmap_sz_bits,
                                                        false,
                                                        device_type,
                                                        sub_bitmap_count};
    if (approx_count_distinct_desc.bitmapPaddedSizeBytes() >=
        precise_count_distinct_desc.bitmapPaddedSizeBytes()) {
      auto precise_count_distinct = hdk::ir::makeExpr<hdk::ir::AggExpr>(
          get_agg_type(hdk::ir::AggType::kCount,
                       arg.get(),
                       executor->getConfig().exec.group_by.bigint_count),
          hdk::ir::AggType::kCount,
          arg,
          true,
          nullptr);
      target_exprs_owned.push_back(precise_count_distinct);
      ra_exe_unit.target_exprs[i] = precise_count_distinct.get();
    }
  }
  return ra_exe_unit;
}

// Here we try to decide if we should use hash partitioning for groupby execution.
// Partitioned aggregation is beneficial in case of a very large output buffer.
// Partitioning allows us to reduce the size of the buffer used for each partition
// and also completely avoid reduction.
// Current heuristic is very simple and covers the case when a lot of very small
// groups are processed.
void maybeRequestPartitionedAggregation(const RelAlgExecutionUnit& ra_exe_unit,
                                        size_t estimated_buffer_entries,
                                        const CompilationOptions& co,
                                        DataProvider* data_provider,
                                        const Config& config) {
  // Partitioned aggregation is supported for CPU only.
  if (co.device_type != ExecutorDeviceType::CPU) {
    VLOG(1) << "Ignore partitioned aggregation option for non-CPU device";
    return;
  }

  // Check if allowed by the config.
  if (!config.exec.group_by.enable_cpu_partitioned_groupby) {
    VLOG(1) << "Ignore partitioned aggregation option. Disabled by config.";
    return;
  }

  for (auto& input : ra_exe_unit.input_col_descs) {
    if (input->type()->isVarLen()) {
      VLOG(1) << "Drop partitioned aggregation option due to varlen data.";
      return;
    }
  }

  // Check output buffer size threshold.
  size_t entry_size = 0;
  for (auto& key : ra_exe_unit.groupby_exprs) {
    if (!key) {
      entry_size += 8;
    } else {
      if (key->type()->isString() || key->type()->isArray()) {
        entry_size += 16;
      } else {
        entry_size += key->type()->canonicalSize();
      }
    }
  }
  for (auto expr : ra_exe_unit.target_exprs) {
    // Skip key columns.
    if (!expr->is<hdk::ir::ColumnVar>()) {
      entry_size += expr->type()->canonicalSize();
    }
  }
  if (estimated_buffer_entries * entry_size <
      config.exec.group_by.partitioning_buffer_size_threshold) {
    VLOG(1)
        << "Drop partitioned aggregation option due to the small output buffer size of "
        << (estimated_buffer_entries * entry_size) << " bytes. Threshold value is "
        << config.exec.group_by.partitioning_buffer_size_threshold;
    return;
  }

  // Request shuffle when the number of estimated groups is comparable to the input
  // rows count. For now, ignore the fact that joins and filters change the number of
  // rows to be aggregated, so simply use the size of the outermost table.
  // Number entries is 2x of the estimated number of groups. Use partitioning if we
  // expect less than the configured threshold number of rows per each group in average.
  auto table_meta =
      data_provider->getTableMetadata(ra_exe_unit.input_descs[0].getDatabaseId(),
                                      ra_exe_unit.input_descs[0].getTableId());
  if (table_meta->getNumTuples() * 2 <
      estimated_buffer_entries * config.exec.group_by.partitioning_group_size_threshold) {
    LOG(INFO) << "Requesting partitioned aggregation (entries="
              << estimated_buffer_entries << ", rows=" << table_meta->getNumTuples()
              << ")";
    throw RequestPartitionedAggregation(entry_size, estimated_buffer_entries);
  }

  VLOG(1) << "Drop partitioned aggregation option due to the small estimated number of "
             "groups. Input rows: "
          << table_meta->getNumTuples()
          << " Estimated group count: " << (estimated_buffer_entries / 2)
          << " Threshold ratio: "
          << config.exec.group_by.partitioning_group_size_threshold;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeWorkUnit(
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<hdk::ir::TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co_in,
    const ExecutionOptions& eo_in,
    const int64_t queue_time_ms,
    const std::optional<size_t> previous_count) {
  INJECT_TIMER(executeWorkUnit);
  auto timer = DEBUG_TIMER(__func__);

  auto co = co_in;
  auto eo = eo_in;
  ColumnCacheMap column_cache;
  if (is_window_execution_unit(work_unit.exe_unit)) {
    if (!config_.exec.window_func.enable) {
      throw std::runtime_error("Window functions support is disabled");
    }
    co.device_type = ExecutorDeviceType::CPU;
    co.allow_lazy_fetch = false;
    computeWindow(work_unit.exe_unit, co, eo, column_cache, queue_time_ms);
  }
  if (!eo.just_explain && eo.find_push_down_candidates) {
    // find potential candidates:
    auto selected_filters = selectFiltersToBePushedDown(work_unit, co, eo);
    if (!selected_filters.empty() || eo.just_calcite_explain) {
      return ExecutionResult(selected_filters, eo.find_push_down_candidates);
    }
  }
  const auto body = work_unit.body;
  CHECK(body);
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

  auto ra_exe_unit = decide_approx_count_distinct_implementation(
      work_unit.exe_unit, table_infos, executor_, co.device_type, target_exprs_owned_);

  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  if (is_window_execution_unit(ra_exe_unit)) {
    CHECK_EQ(table_infos.size(), size_t(1));
    CHECK_EQ(table_infos.front().info->fragments.size(), size_t(1));
    max_groups_buffer_entry_guess =
        table_infos.front().info->fragments.front().getNumTuples();
    ra_exe_unit.scan_limit = max_groups_buffer_entry_guess;
  } else if (compute_output_buffer_size(ra_exe_unit) && !isRowidLookup(work_unit)) {
    if (previous_count && !exe_unit_has_quals(ra_exe_unit)) {
      ra_exe_unit.scan_limit = *previous_count;
    } else {
      if (eo.executor_type == ::ExecutorType::Extern) {
        ra_exe_unit.scan_limit = 0;
      } else if (!eo.just_explain) {
        const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
        if (filter_count_all) {
          ra_exe_unit.scan_limit = std::max(*filter_count_all, size_t(1));
        }
      }
    }
  }

  if (g_columnar_large_projections) {
    const auto prefer_columnar = should_output_columnar(ra_exe_unit);
    if (prefer_columnar) {
      VLOG(1) << "Using columnar layout for projection as output size of "
              << ra_exe_unit.scan_limit << " rows exceeds threshold of "
              << g_columnar_large_projections_threshold << ".";
      eo.output_columnar_hint = true;
    }
  }

  ExecutionResult result;
  auto execute_and_handle_errors = [&](const auto max_groups_buffer_entry_guess_in,
                                       const bool has_cardinality_estimation,
                                       const bool has_ndv_estimation) -> ExecutionResult {
    // Note that the groups buffer entry guess may be modified during query execution.
    // Create a local copy so we can track those changes if we need to attempt a retry
    // due to OOM
    auto local_groups_buffer_entry_guess = max_groups_buffer_entry_guess_in;
    try {
      auto rs_table = executor_->executeWorkUnit(local_groups_buffer_entry_guess,
                                                 is_agg,
                                                 table_infos,
                                                 ra_exe_unit,
                                                 co,
                                                 eo,
                                                 has_cardinality_estimation,
                                                 data_provider_,
                                                 column_cache);
      rs_table.setQueueTime(queue_time_ms);
      return registerResultSetTable(rs_table, targets_meta, eo.just_explain);
    } catch (const QueryExecutionError& e) {
      if (!has_ndv_estimation && e.getErrorCode() < 0) {
        throw CardinalityEstimationRequired(/*range=*/0);
      }
      handlePersistentError(e.getErrorCode());
      return handleOutOfMemoryRetry(
          {ra_exe_unit, work_unit.body, local_groups_buffer_entry_guess},
          targets_meta,
          is_agg,
          co,
          eo,
          e.wasMultifragKernelLaunch(),
          queue_time_ms);
    }
  };

  auto cache_key = ra_exec_unit_desc_for_caching(ra_exe_unit);
  try {
    auto cached_cardinality = executor_->getCachedCardinality(cache_key);
    auto card = cached_cardinality.second;
    if (cached_cardinality.first && card >= 0) {
      result = execute_and_handle_errors(
          card, /*has_cardinality_estimation=*/true, /*has_ndv_estimation=*/false);
    } else {
      result = execute_and_handle_errors(
          max_groups_buffer_entry_guess,
          // In case of partitioned aggregation, we use max partition size as an
          // estimation.
          // In case of small enough input, we use configured default estimation.
          // Otherwise, we state that we don't have an estimation and cardinality
          // estimator would be requested in case of baseline hash groupby.
          ra_exe_unit.partitioned_aggregation ||
              groups_approx_upper_bound(table_infos) <=
                  config_.exec.group_by.big_group_threshold,
          /*has_ndv_estimation=*/ra_exe_unit.partitioned_aggregation);
    }
  } catch (const CardinalityEstimationRequired& e) {
    // check the cardinality cache
    auto cached_cardinality = executor_->getCachedCardinality(cache_key);
    auto card = cached_cardinality.second;
    if (cached_cardinality.first && card >= 0) {
      result = execute_and_handle_errors(card, true, /*has_ndv_estimation=*/true);
    } else {
      const auto ndv_groups_estimation =
          getNDVEstimation(work_unit, e.range(), is_agg, co, eo);
      const auto estimated_groups_buffer_entry_guess =
          ndv_groups_estimation > 0 ? 2 * ndv_groups_estimation
                                    : std::min(groups_approx_upper_bound(table_infos),
                                               g_estimator_failure_max_groupby_size);
      CHECK_GT(estimated_groups_buffer_entry_guess, size_t(0));
      maybeRequestPartitionedAggregation(
          ra_exe_unit, estimated_groups_buffer_entry_guess, co, data_provider_, config_);
      result = execute_and_handle_errors(
          estimated_groups_buffer_entry_guess, true, /*has_ndv_estimation=*/true);
      if (!(eo.just_validate || eo.just_explain)) {
        executor_->addToCardinalityCache(cache_key, estimated_groups_buffer_entry_guess);
      }
    }
  }

  return result;
}

std::optional<size_t> RelAlgExecutor::getFilteredCountAll(const WorkUnit& work_unit,
                                                          const bool is_agg,
                                                          const CompilationOptions& co,
                                                          const ExecutionOptions& eo) {
  auto unnested_vars = UnnestedVarsCollector::collect(work_unit.exe_unit.target_exprs);
  hdk::ir::QueryBuilder builder(
      hdk::ir::Context::defaultCtx(), schema_provider_, executor_->getConfigPtr());
  hdk::ir::BuilderExpr count_all_agg;
  if (!unnested_vars.empty()) {
    hdk::ir::BuilderExpr total_count;
    for (auto var : unnested_vars) {
      hdk::ir::BuilderExpr var_expr(&builder, var->shared());
      if (!total_count.expr()) {
        total_count = var_expr.cardinality();
      } else {
        total_count = total_count.mul(var_expr.cardinality());
      }
    }
    count_all_agg = total_count.sum();
  } else {
    count_all_agg = builder.count();
  }

  const auto count_all_exe_unit = create_count_all_execution_unit(
      work_unit.exe_unit, schema_provider_.get(), count_all_agg.expr());
  size_t one{1};
  hdk::ResultSetTable count_all_result;
  try {
    ColumnCacheMap column_cache;
    count_all_result =
        executor_->executeWorkUnit(one,
                                   is_agg,
                                   get_table_infos(work_unit.exe_unit, executor_),
                                   count_all_exe_unit,
                                   co,
                                   eo,
                                   false,
                                   data_provider_,
                                   column_cache);
  } catch (const QueryMustRunOnCpu&) {
    // force a retry of the top level query on CPU
    throw;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to run pre-flight filtered count with error " << e.what();
    return std::nullopt;
  }
  CHECK_EQ(count_all_result.size(), (size_t)1);
  const auto count_row = count_all_result[0]->getNextRow(false, false);
  CHECK_EQ(size_t(1), count_row.size());
  const auto& count_tv = count_row.front();
  const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
  CHECK(count_scalar_tv);
  const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
  CHECK(count_ptr);
  CHECK_GE(*count_ptr, 0);
  auto count_upper_bound = static_cast<size_t>(*count_ptr);
  return std::max(count_upper_bound, size_t(1));
}

bool RelAlgExecutor::isRowidLookup(const WorkUnit& work_unit) {
  const auto& ra_exe_unit = work_unit.exe_unit;
  if (ra_exe_unit.input_descs.size() != 1) {
    return false;
  }
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
    const auto comp_expr = std::dynamic_pointer_cast<const hdk::ir::BinOper>(simple_qual);
    if (!comp_expr || !comp_expr->isEq()) {
      return false;
    }
    const auto lhs = comp_expr->leftOperand();
    const auto lhs_col = dynamic_cast<const hdk::ir::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->tableId() || lhs_col->rteIdx()) {
      return false;
    }
    const auto rhs = comp_expr->rightOperand();
    const auto rhs_const = dynamic_cast<const hdk::ir::Constant*>(rhs);
    if (!rhs_const) {
      return false;
    }
    return lhs_col->isVirtual();
  }
  return false;
}

ExecutionResult RelAlgExecutor::handleOutOfMemoryRetry(
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<hdk::ir::TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const bool was_multifrag_kernel_launch,
    const int64_t queue_time_ms) {
  // Disable the bump allocator
  // Note that this will have basically the same affect as using the bump allocator
  // for the kernel per fragment path. Need to unify the max_groups_buffer_entry_guess
  // = 0 path and the bump allocator path for kernel per fragment execution.
  auto ra_exe_unit_in = work_unit.exe_unit;

  hdk::ResultSetTable result;
  const auto table_infos = get_table_infos(ra_exe_unit_in, executor_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  const ExecutionOptions eo_no_multifrag = [&]() {
    ExecutionOptions copy = eo;
    copy.allow_multifrag = false;
    copy.just_explain = false;
    copy.find_push_down_candidates = false;
    copy.just_calcite_explain = false;
    return copy;
  }();

  if (was_multifrag_kernel_launch) {
    try {
      // Attempt to retry using the kernel per fragment path. The smaller input size
      // required may allow the entire kernel to execute in GPU memory.
      LOG(WARNING) << "Multifrag query ran out of memory, retrying with multifragment "
                      "kernels disabled.";
      const auto ra_exe_unit = decide_approx_count_distinct_implementation(
          ra_exe_unit_in, table_infos, executor_, co.device_type, target_exprs_owned_);
      ColumnCacheMap column_cache;
      result = executor_->executeWorkUnit(max_groups_buffer_entry_guess,
                                          is_agg,
                                          table_infos,
                                          ra_exe_unit,
                                          co,
                                          eo_no_multifrag,
                                          true,
                                          data_provider_,
                                          column_cache);
    } catch (const QueryExecutionError& e) {
      handlePersistentError(e.getErrorCode());
      LOG(WARNING) << "Kernel per fragment query ran out of memory, retrying on CPU.";
    }
  }

  const auto co_cpu = CompilationOptions::makeCpuOnly(co);
  // Only reset the group buffer entry guess if we ran out of slots, which
  // suggests a
  // highly pathological input which prevented a good estimation of distinct tuple
  // count. For projection queries, this will force a per-fragment scan limit, which
  // is compatible with the CPU path
  VLOG(1) << "Resetting max groups buffer entry guess.";
  max_groups_buffer_entry_guess = 0;

  int iteration_ctr = -1;
  while (!result.empty()) {
    iteration_ctr++;
    auto ra_exe_unit = decide_approx_count_distinct_implementation(
        ra_exe_unit_in, table_infos, executor_, co_cpu.device_type, target_exprs_owned_);
    ColumnCacheMap column_cache;
    try {
      result = executor_->executeWorkUnit(max_groups_buffer_entry_guess,
                                          is_agg,
                                          table_infos,
                                          ra_exe_unit,
                                          co_cpu,
                                          eo_no_multifrag,
                                          true,
                                          data_provider_,
                                          column_cache);
    } catch (const QueryExecutionError& e) {
      // Ran out of slots
      if (e.getErrorCode() < 0) {
        // Even the conservative guess failed; it should only happen when we group
        // by a huge cardinality array. Maybe we should throw an exception instead?
        // Such a heavy query is entirely capable of exhausting all the host memory.
        CHECK(max_groups_buffer_entry_guess);
        // Only allow two iterations of increasingly large entry guesses up to a
        // maximum of 512MB per column per kernel
        if (config_.exec.watchdog.enable || iteration_ctr > 1) {
          throw std::runtime_error("Query ran out of output slots in the result");
        }
        max_groups_buffer_entry_guess *= 2;
        LOG(WARNING) << "Query ran out of slots in the output buffer, retrying with max "
                        "groups buffer entry "
                        "guess equal to "
                     << max_groups_buffer_entry_guess;
      } else {
        handlePersistentError(e.getErrorCode());
      }
    }
  }
  result.setQueueTime(queue_time_ms);
  return registerResultSetTable(result, targets_meta, eo.just_explain);
}

void RelAlgExecutor::handlePersistentError(const int32_t error_code) {
  LOG(ERROR) << "Query execution failed with error "
             << getErrorMessageFromCode(error_code);
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    // We ran out of GPU memory, this doesn't count as an error if the query is
    // allowed to continue on CPU because retry on CPU is explicitly allowed through
    // --allow-cpu-retry.
    LOG(INFO) << "Query ran out of GPU memory, attempting punt to CPU";
    if (!config_.exec.heterogeneous.allow_cpu_retry) {
      throw std::runtime_error(
          "Query ran out of GPU memory, unable to automatically retry on CPU");
    }
    return;
  }
  throw std::runtime_error(getErrorMessageFromCode(error_code));
}

ExecutionResult RelAlgExecutor::registerResultSetTable(
    hdk::ResultSetTable table,
    const std::vector<hdk::ir::TargetMetaInfo>& targets_meta,
    bool just_explain_result) {
  std::vector<std::string> col_names;
  if (just_explain_result) {
    col_names.push_back("explain");
  } else {
    col_names.reserve(targets_meta.size());
    for (auto& meta : targets_meta) {
      col_names.emplace_back(meta.get_resname());
    }
  }

  CHECK(!table.empty());
  table[0]->setColNames(std::move(col_names));
  auto token = rs_registry_->put(std::move(table));
  return {token, targets_meta};
}

namespace {
struct ErrorInfo {
  const char* code{nullptr};
  const char* description{nullptr};
};
ErrorInfo getErrorDescription(const int32_t error_code) {
  switch (error_code) {
    case Executor::ERR_DIV_BY_ZERO:
      return {"ERR_DIV_BY_ZERO", "Division by zero"};
    case Executor::ERR_OUT_OF_GPU_MEM:
      return {"ERR_OUT_OF_GPU_MEM",

              "Query couldn't keep the entire working set of columns in GPU memory"};
    case Executor::ERR_UNSUPPORTED_SELF_JOIN:
      return {"ERR_UNSUPPORTED_SELF_JOIN", "Self joins not supported yet"};
    case Executor::ERR_OUT_OF_CPU_MEM:
      return {"ERR_OUT_OF_CPU_MEM", "Not enough host memory to execute the query"};
    case Executor::ERR_OVERFLOW_OR_UNDERFLOW:
      return {"ERR_OVERFLOW_OR_UNDERFLOW", "Overflow or underflow"};
    case Executor::ERR_OUT_OF_TIME:
      return {"ERR_OUT_OF_TIME", "Query execution has exceeded the time limit"};
    case Executor::ERR_INTERRUPTED:
      return {"ERR_INTERRUPTED", "Query execution has been interrupted"};
    case Executor::ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED:
      return {"ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED",
              "Columnar conversion not supported for variable length types"};
    case Executor::ERR_TOO_MANY_LITERALS:
      return {"ERR_TOO_MANY_LITERALS", "Too many literals in the query"};
    case Executor::ERR_STRING_CONST_IN_RESULTSET:
      return {"ERR_STRING_CONST_IN_RESULTSET",
              "NONE ENCODED String types are not supported as input result set."};
    case Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES:
      return {"ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES",
              "Multiple distinct values encountered"};
    case Executor::ERR_WIDTH_BUCKET_INVALID_ARGUMENT:
      return {"ERR_WIDTH_BUCKET_INVALID_ARGUMENT",
              "Arguments of WIDTH_BUCKET function does not satisfy the condition"};
    default:
      return {nullptr, nullptr};
  }
}

}  // namespace

std::string RelAlgExecutor::getErrorMessageFromCode(const int32_t error_code) {
  if (error_code < 0) {
    return "Ran out of slots in the query output buffer";
  }
  const auto errorInfo = getErrorDescription(error_code);

  if (errorInfo.code) {
    return errorInfo.code + ": "s + errorInfo.description;
  } else {
    return "Other error: code "s + std::to_string(error_code);
  }
}

void RelAlgExecutor::executePostExecutionCallback() {
  if (post_execution_callback_) {
    VLOG(1) << "Running post execution callback.";
    (*post_execution_callback_)();
  }
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const hdk::ir::Node* node,
                                                        const CompilationOptions& co,
                                                        const ExecutionOptions& eo,
                                                        bool allow_speculative_sort) {
  hdk::WorkUnitBuilder builder(node,
                               query_dag_.get(),
                               executor_,
                               schema_provider_,
                               temporary_tables_,
                               eo,
                               co,
                               now_,
                               false,
                               allow_speculative_sort);
  auto exe_unit = builder.exeUnit();
  const auto query_infos = get_table_infos(exe_unit.input_descs, executor_);
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta = get_targets_meta(node, rewritten_exe_unit.target_exprs);
  node->setOutputMetainfo(targets_meta);
  RelAlgTranslator translator(
      executor_, builder.nestLevels(), builder.joinTypes(), now_, eo.just_explain);
  auto& left_deep_trees_info = getLeftDeepJoinTreesInfo();
  std::optional<unsigned> left_deep_tree_id = builder.leftDeepTreeId();
  if (left_deep_tree_id && left_deep_tree_id.has_value()) {
    left_deep_trees_info.emplace(left_deep_tree_id.value(),
                                 rewritten_exe_unit.join_quals);
  }
  auto dag_info = QueryPlanDagExtractor::extractQueryPlanDag(node,
                                                             schema_provider_,
                                                             left_deep_tree_id,
                                                             left_deep_trees_info,
                                                             temporary_tables_,
                                                             executor_,
                                                             translator);
  if (is_extracted_dag_valid(dag_info)) {
    rewritten_exe_unit.query_plan_dag = dag_info.extracted_dag;
    rewritten_exe_unit.hash_table_build_plan_dag = dag_info.hash_table_plan_dag;
    rewritten_exe_unit.table_id_to_node_map = dag_info.table_id_to_node_map;
  }

  target_exprs_owned_ = builder.releaseTargetExprsOwned();
  rewritten_exe_unit.cost_model = executor_->getCostModel();

  return {rewritten_exe_unit,
          node,
          builder.maxGroupsBufferEntryGuess(),
          std::move(query_rewriter),
          builder.releaseInputPermutation(),
          builder.releaseLeftDeepJoinInputSizes()};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const hdk::ir::Node* node,
                                                        const SortInfo& sort_info,
                                                        const ExecutionOptions& eo) {
  CHECK(sort_info.order_entries.empty());
  return createWorkUnit(node, CompilationOptions::defaults(), eo, true);
}

namespace {

JoinType get_join_type(const hdk::ir::Node* ra) {
  auto sink = get_data_sink(ra);
  if (auto join = dynamic_cast<const hdk::ir::Join*>(sink)) {
    return join->getJoinType();
  }
  return JoinType::INVALID;
}

}  // namespace

std::shared_ptr<RelAlgTranslator> RelAlgExecutor::getRelAlgTranslator(
    const hdk::ir::Node* node) {
  auto input_to_nest_level = get_input_nest_levels(node, {});
  const auto join_types = std::vector<JoinType>{get_join_type(node)};
  return std::make_shared<RelAlgTranslator>(
      executor_, input_to_nest_level, join_types, now_, false);
}

SpeculativeTopNBlacklist RelAlgExecutor::speculative_topn_blacklist_;
