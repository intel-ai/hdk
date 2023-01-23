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

#include "QueryPhysicalInputsCollector.h"

#include "ExprDagVisitor.h"
#include "RelAlgDagBuilder.h"
#include "RelAlgVisitor.h"

#include "SchemaMgr/ColumnInfo.h"

namespace {

using InputColDescriptorSet = std::unordered_set<InputColDescriptor>;

template <typename ExprVisitor, typename ResultType>
class PhysicalInputsNodeVisitor : public RelAlgVisitor<ResultType> {
 public:
  PhysicalInputsNodeVisitor() {}

  using RelAlgVisitor<ResultType>::visit;

  ResultType visitCompound(const hdk::ir::Compound* compound) const override {
    ResultType result;
    ExprVisitor visitor;
    for (auto& expr : compound->getGroupByExprs()) {
      auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    for (auto& expr : compound->getExprs()) {
      auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    auto filter = compound->getFilter();
    if (filter) {
      auto inputs = visitor.visit(filter.get());
      result.insert(inputs.begin(), inputs.end());
    }
    return result;
  }

  ResultType visitFilter(const hdk::ir::Filter* filter) const override {
    ExprVisitor visitor;
    return visitor.visit(filter->getConditionExpr());
  }

  ResultType visitJoin(const hdk::ir::Join* join) const override {
    auto condition = join->getCondition();
    if (!condition) {
      return ResultType{};
    }
    ExprVisitor visitor;
    return visitor.visit(condition);
  }

  ResultType visitLeftDeepInnerJoin(
      const hdk::ir::LeftDeepInnerJoin* left_deep_inner_join) const override {
    ResultType result;
    auto condition = left_deep_inner_join->getInnerCondition();
    ExprVisitor visitor;
    if (condition) {
      result = visitor.visit(condition);
    }
    for (size_t nesting_level = 1; nesting_level < left_deep_inner_join->inputCount();
         ++nesting_level) {
      auto outer_condition = left_deep_inner_join->getOuterCondition(nesting_level);
      if (outer_condition) {
        auto outer_result = visitor.visit(outer_condition);
        result.insert(outer_result.begin(), outer_result.end());
      }
    }
    return result;
  }

  ResultType visitProject(const hdk::ir::Project* project) const override {
    ResultType result;
    ExprVisitor visitor;
    for (auto& expr : project->getExprs()) {
      const auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    return result;
  }

  ResultType visitSort(const hdk::ir::Sort* sort) const override {
    return this->visit(sort->getInput(0));
  }

  ResultType visitAggregate(const hdk::ir::Aggregate* agg) const override {
    ResultType result;
    ExprVisitor visitor;
    for (auto& expr : agg->getAggs()) {
      const auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    for (unsigned i = 0; i < agg->getGroupByCount(); ++i) {
      const auto inputs = visitor.visit(getNodeColumnRef(agg->getInput(0), i).get());
      result.insert(inputs.begin(), inputs.end());
    }
    return result;
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};
template <typename Derived, typename ResultType>
class InputVisitorBase : public ScalarExprVisitor<ResultType> {
 public:
  InputVisitorBase() {}

  ResultType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    PhysicalInputsNodeVisitor<Derived, ResultType> visitor;
    return visitor.visit(subquery->node());
  }

  ResultType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    PhysicalInputsNodeVisitor<Derived, ResultType> visitor;
    auto node_res = visitor.visit(in_subquery->node());
    auto arg_res = ScalarExprVisitor<ResultType>::visit(in_subquery->arg());
    return aggregateResult(node_res, arg_res);
  }

  ResultType visitWindowFunction(
      const hdk::ir::WindowFunction* window_func) const override {
    ResultType result;
    for (auto& part_key : window_func->partitionKeys()) {
      if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(part_key.get())) {
        if (auto filter = dynamic_cast<const hdk::ir::Filter*>(col_ref->node())) {
          // Partitions utilize string dictionary translation in the hash join framework
          // if the partition key is a dictionary encoded string. Ensure we reach the
          // source for all partition keys, so we can access string dictionaries for the
          // partition keys while we build the partition (hash) table
          auto parent_node = filter->getInput(0);
          auto input = getNodeColumnRef(parent_node, col_ref->index());
          result = aggregateResult(result, this->visit(input.get()));
        }
        result = aggregateResult(result, this->visitColumnRef(col_ref));
      }
    }
    for (const auto& arg : window_func->args()) {
      result = aggregateResult(result, this->visit(arg.get()));
    }
    // TODO: order keys are ignored here to get the same result as for Rex.
    // Should it be fixed to collect proper inputs?
    return result;
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class PhysicalInputsVisitor
    : public InputVisitorBase<PhysicalInputsVisitor, InputColDescriptorSet> {
 public:
  PhysicalInputsVisitor() {}

  InputColDescriptorSet visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    const auto source = col_ref->node();
    const auto scan = dynamic_cast<const hdk::ir::Scan*>(source);
    if (!scan) {
      const auto join = dynamic_cast<const hdk::ir::Join*>(source);
      if (join) {
        auto input = getJoinInputColumnRef(col_ref);
        return visit(input.get());
      }
      // Filter and sort indirectly use all columns of its input.
      // So, walk through such column references.
      if (source->is<hdk::ir::Filter>() || source->is<hdk::ir::Scan>()) {
        auto new_ref = getNodeColumnRef(source->getInput(0), col_ref->index());
        return visit(new_ref.get());
      }
      // Aggregate implicitly uses group columns.
      if (auto agg = source->as<hdk::ir::Aggregate>()) {
        if (col_ref->index() < agg->getGroupByCount()) {
          auto new_ref = getNodeColumnRef(agg->getInput(0), col_ref->index());
          return visit(new_ref.get());
        }
      }
      return InputColDescriptorSet{};
    }

    auto col_info = scan->getColumnInfo(col_ref->index());
    CHECK_GT(col_info->table_id, 0);
    return {{col_info, 0}};
  }
};

template <typename RelAlgVisitor, typename ResultType>
class SubqueryVisitorBase : public ScalarExprVisitor<ResultType> {
 public:
  SubqueryVisitorBase() {}

  ResultType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    RelAlgVisitor visitor;
    return visitor.visit(subquery->node());
  }

  ResultType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    RelAlgVisitor visitor;
    return visitor.visit(in_subquery->node());
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class PhysicalColumnInfosNodeVisitor
    : public PhysicalInputsNodeVisitor<
          SubqueryVisitorBase<PhysicalColumnInfosNodeVisitor, ColumnInfoMap>,
          ColumnInfoMap> {
 public:
  PhysicalColumnInfosNodeVisitor() {}

  ColumnInfoMap visitScan(const hdk::ir::Scan* scan) const override {
    ColumnInfoMap res;

    for (size_t col_idx = 0; col_idx < scan->size(); ++col_idx) {
      auto col_info = scan->getColumnInfo(col_idx);
      res.insert({*col_info, col_info});
    }

    return res;
  }
};

class PhysicalTableInputsNodeVisitor : public ExprDagVisitor {
 public:
  PhysicalTableInputsNodeVisitor() {}

  using ExprDagVisitor::visit;
  using TableIds = std::unordered_set<std::pair<int, int>>;

  static TableIds getTableIds(hdk::ir::Node const* node) {
    PhysicalTableInputsNodeVisitor visitor;
    visitor.visit(node);
    return std::move(visitor.table_ids_);
  }

 private:
  TableIds table_ids_;

  void visitScan(const hdk::ir::Scan* scan) override {
    table_ids_.insert({scan->getDatabaseId(), scan->getTableId()});
  }
};

class PhysicalTableInfosNodeVisitor
    : public PhysicalInputsNodeVisitor<
          SubqueryVisitorBase<PhysicalTableInfosNodeVisitor, TableInfoMap>,
          TableInfoMap> {
 public:
  PhysicalTableInfosNodeVisitor() {}

  TableInfoMap visitScan(const hdk::ir::Scan* scan) const override {
    TableInfoMap res;
    auto info = scan->getTableInfo();
    res.insert(std::make_pair(TableRef(info->db_id, info->table_id), info));
    return res;
  }
};

}  // namespace

std::unordered_set<InputColDescriptor> get_physical_inputs(const hdk::ir::Node* ra) {
  PhysicalInputsNodeVisitor<PhysicalInputsVisitor, InputColDescriptorSet>
      phys_inputs_visitor;
  return phys_inputs_visitor.visit(ra);
}

std::unordered_set<std::pair<int, int>> get_physical_table_inputs(
    const hdk::ir::Node* ra) {
  return PhysicalTableInputsNodeVisitor::getTableIds(ra);
}

ColumnInfoMap get_physical_column_infos(const hdk::ir::Node* ra) {
  PhysicalColumnInfosNodeVisitor visitor;
  return visitor.visit(ra);
}

TableInfoMap get_physical_table_infos(const hdk::ir::Node* ra) {
  PhysicalTableInfosNodeVisitor visitor;
  return visitor.visit(ra);
}

std::ostream& operator<<(std::ostream& os, PhysicalInput const& physical_input) {
  return os << '(' << physical_input.col_id << ',' << physical_input.table_id << ')';
}
