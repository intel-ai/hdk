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

#include "TestRelAlgDagBuilder.h"

#include "QueryEngine/RelAlgTranslator.h"

void TestRelAlgDagBuilder::addNode(hdk::ir::NodePtr node) {
  nodes_.push_back(node);
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addScan(const TableRef& table) {
  return addScan(schema_provider_->getTableInfo(table));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addScan(int db_id, int table_id) {
  return addScan(schema_provider_->getTableInfo(db_id, table_id));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addScan(int db_id, const std::string& table_name) {
  return addScan(schema_provider_->getTableInfo(db_id, table_name));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addScan(TableInfoPtr table_info) {
  auto col_infos = schema_provider_->listColumns(*table_info);
  auto scan_node = std::make_shared<hdk::ir::Scan>(table_info, std::move(col_infos));
  nodes_.push_back(scan_node);
  return scan_node;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addProject(hdk::ir::NodePtr input,
                                                  const std::vector<std::string>& fields,
                                                  const std::vector<int>& cols) {
  hdk::ir::ExprPtrVector exprs;
  auto input_cols = getNodeColumnRefs(input.get());
  for (auto col_idx : cols) {
    CHECK_LT((size_t)col_idx, input_cols.size());
    exprs.push_back(input_cols[col_idx]);
  }
  return addProject(input, fields, std::move(exprs));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addProject(hdk::ir::NodePtr input,
                                                  const std::vector<std::string>& fields,
                                                  hdk::ir::ExprPtrVector exprs) {
  auto res = std::make_shared<hdk::ir::Project>(std::move(exprs), fields, input);
  nodes_.push_back(res);
  return res;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addProject(hdk::ir::NodePtr input,
                                                  const std::vector<int>& cols) {
  auto fields = buildFieldNames(cols.size());
  return addProject(input, fields, std::move(cols));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addProject(hdk::ir::NodePtr input,
                                                  hdk::ir::ExprPtrVector exprs) {
  auto fields = buildFieldNames(exprs.size());
  return addProject(input, fields, std::move(exprs));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addFilter(hdk::ir::NodePtr input,
                                                 hdk::ir::ExprPtr expr) {
  auto res = std::make_shared<hdk::ir::Filter>(std::move(expr), std::move(input));
  nodes_.push_back(res);
  return res;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addAgg(hdk::ir::NodePtr input,
                                              const std::vector<std::string>& fields,
                                              size_t group_size,
                                              hdk::ir::ExprPtrVector aggs) {
  auto res =
      std::make_shared<hdk::ir::Aggregate>(group_size, std::move(aggs), fields, input);
  nodes_.push_back(res);
  return res;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addAgg(hdk::ir::NodePtr input,
                                              const std::vector<std::string>& fields,
                                              size_t group_size,
                                              std::vector<AggDesc> aggs) {
  hdk::ir::ExprPtrVector agg_exprs;
  for (auto& agg : aggs) {
    hdk::ir::ExprPtr arg_expr;
    CHECK_LE(agg.operands.size(), size_t(1));
    if (agg.operands.size()) {
      arg_expr = hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          getColumnType(input.get(), agg.operands[0]), input.get(), agg.operands[0]);
    }
    agg_exprs.push_back(hdk::ir::makeExpr<hdk::ir::AggExpr>(
        agg.type, agg.agg, arg_expr, agg.distinct, nullptr));
  }
  return addAgg(input, fields, group_size, std::move(agg_exprs));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addAgg(hdk::ir::NodePtr input,
                                              size_t group_size,
                                              hdk::ir::ExprPtrVector aggs) {
  auto fields = buildFieldNames(group_size + aggs.size());
  return addAgg(input, fields, group_size, std::move(aggs));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addAgg(hdk::ir::NodePtr input,
                                              size_t group_size,
                                              std::vector<AggDesc> aggs) {
  auto fields = buildFieldNames(group_size + aggs.size());
  return addAgg(input, fields, group_size, std::move(aggs));
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addSort(
    hdk::ir::NodePtr input,
    const std::vector<hdk::ir::SortField>& collation,
    const size_t limit,
    const size_t offset) {
  auto res = std::make_shared<hdk::ir::Sort>(collation, limit, offset, input);
  nodes_.push_back(res);
  return res;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addJoin(hdk::ir::NodePtr lhs,
                                               hdk::ir::NodePtr rhs,
                                               const JoinType join_type,
                                               hdk::ir::ExprPtr condition) {
  auto res = std::make_shared<hdk::ir::Join>(lhs, rhs, std::move(condition), join_type);
  nodes_.push_back(res);
  return res;
}

hdk::ir::NodePtr TestRelAlgDagBuilder::addEquiJoin(hdk::ir::NodePtr lhs,
                                                   hdk::ir::NodePtr rhs,
                                                   const JoinType join_type,
                                                   size_t lhs_col_idx,
                                                   size_t rhs_col_idx) {
  auto lhs_expr = hdk::ir::makeExpr<hdk::ir::ColumnRef>(
      getColumnType(lhs.get(), lhs_col_idx), lhs.get(), lhs_col_idx);
  auto rhs_expr = hdk::ir::makeExpr<hdk::ir::ColumnRef>(
      getColumnType(rhs.get(), rhs_col_idx), rhs.get(), rhs_col_idx);
  auto eq_expr = hdk::ir::makeExpr<hdk::ir::BinOper>(lhs_expr->ctx().boolean(),
                                                     hdk::ir::OpType::kEq,
                                                     hdk::ir::Qualifier::kOne,
                                                     lhs_expr,
                                                     rhs_expr);
  return addJoin(lhs, rhs, join_type, std::move(eq_expr));
}

std::vector<std::string> TestRelAlgDagBuilder::buildFieldNames(size_t count) const {
  std::vector<std::string> res;
  for (size_t i = 1; i <= count; ++i) {
    res.push_back(std::string("field_") + std::to_string(i));
  }
  return res;
}

void TestRelAlgDagBuilder::setRoot(hdk::ir::NodePtr root) {
  root_ = root;
}
