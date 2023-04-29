/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Node.h"
#include "InputRewriter.h"

namespace hdk::ir {

namespace {

const unsigned FIRST_NODE_ID = 1;

std::set<std::pair<const hdk::ir::Node*, int>> getEquivCols(const hdk::ir::Node* node,
                                                            const size_t which_col) {
  std::set<std::pair<const hdk::ir::Node*, int>> work_set;
  auto walker = node;
  auto curr_col = which_col;
  while (true) {
    work_set.insert(std::make_pair(walker, curr_col));
    if (dynamic_cast<const hdk::ir::Scan*>(walker) ||
        dynamic_cast<const hdk::ir::Join*>(walker)) {
      break;
    }
    CHECK_EQ(size_t(1), walker->inputCount());
    auto only_source = walker->getInput(0);
    if (auto project = dynamic_cast<const hdk::ir::Project*>(walker)) {
      if (auto col_ref =
              dynamic_cast<const hdk::ir::ColumnRef*>(project->getExpr(curr_col).get())) {
        const auto join_source = dynamic_cast<const hdk::ir::Join*>(only_source);
        if (join_source) {
          CHECK_EQ(size_t(2), join_source->inputCount());
          auto lhs = join_source->getInput(0);
          CHECK((col_ref->index() < lhs->size() && lhs == col_ref->node()) ||
                join_source->getInput(1) == col_ref->node());
        } else {
          CHECK_EQ(col_ref->node(), only_source);
        }
        curr_col = col_ref->index();
      } else {
        break;
      }
    } else if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(walker)) {
      if (curr_col >= aggregate->getGroupByCount()) {
        break;
      }
    }
    walker = only_source;
  }
  return work_set;
}

template <typename T>
bool is_one_of(const Node* node) {
  return dynamic_cast<const T*>(node);
}

template <typename T1, typename T2, typename... Ts>
bool is_one_of(const Node* node) {
  return dynamic_cast<const T1*>(node) || is_one_of<T2, Ts...>(node);
}

bool isRenamedInput(const Node* node, const size_t index, const std::string& new_name) {
  CHECK_LT(index, node->size());
  if (auto join = dynamic_cast<const Join*>(node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    const auto lhs_size = join->getInput(0)->size();
    if (index < lhs_size) {
      return isRenamedInput(join->getInput(0), index, new_name);
    }
    CHECK_GE(index, lhs_size);
    return isRenamedInput(join->getInput(1), index - lhs_size, new_name);
  }

  if (auto scan = dynamic_cast<const Scan*>(node)) {
    return new_name != scan->getFieldName(index);
  }

  if (auto aggregate = dynamic_cast<const Aggregate*>(node)) {
    return new_name != aggregate->getFieldName(index);
  }

  if (auto project = dynamic_cast<const Project*>(node)) {
    return new_name != project->getFieldName(index);
  }

  if (auto logical_values = dynamic_cast<const LogicalValues*>(node)) {
    const auto& tuple_type = logical_values->getTupleType();
    CHECK_LT(index, tuple_type.size());
    return new_name != tuple_type[index].get_resname();
  }

  CHECK(dynamic_cast<const Sort*>(node) || dynamic_cast<const Filter*>(node) ||
        dynamic_cast<const LogicalUnion*>(node));
  return isRenamedInput(node->getInput(0), index, new_name);
}

}  // namespace

thread_local unsigned Node::crt_id_ = FIRST_NODE_ID;

Node::Node(NodeInputs inputs)
    : inputs_(std::move(inputs))
    , id_(crt_id_++)
    , context_data_(nullptr)
    , is_nop_(false) {}

void Node::replaceInput(
    std::shared_ptr<const Node> old_input,
    std::shared_ptr<const Node> input,
    std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map) {
  InputRewriter rewriter;
  if (old_to_new_index_map) {
    rewriter.addNodeMapping(old_input.get(), input.get(), *old_to_new_index_map);
  } else {
    rewriter.addNodeMapping(old_input.get(), input.get());
  }
  replaceInput(old_input, input, rewriter);
}

void Node::replaceInput(std::shared_ptr<const Node> old_input,
                        std::shared_ptr<const Node> input,
                        hdk::ir::ExprRewriter& input_redirector) {
  bool replaced = false;
  for (auto& input_ptr : inputs_) {
    if (input_ptr == old_input) {
      input_ptr = input;
      replaced = true;
    }
  }
  if (replaced) {
    rewriteExprs(input_redirector);
  }
}

void Node::resetRelAlgFirstId() noexcept {
  crt_id_ = FIRST_NODE_ID;
}

void Node::print() const {
  std::cout << toString() << std::endl;
}

Project::Project(Project const& rhs)
    : Node(rhs), exprs_(rhs.exprs_), fields_(rhs.fields_) {}

void Project::rewriteExprs(hdk::ir::ExprRewriter& rewriter) {
  for (size_t i = 0; i < exprs_.size(); ++i) {
    exprs_[i] = rewriter.visit(exprs_[i].get());
  }
}

void Project::appendInput(std::string new_field_name, ExprPtr expr) {
  fields_.emplace_back(std::move(new_field_name));
  exprs_.emplace_back(std::move(expr));
}

bool Project::isIdentity() const {
  if (!isSimple()) {
    return false;
  }
  CHECK_EQ(size_t(1), inputCount());
  const auto source = getInput(0);
  if (dynamic_cast<const Join*>(source)) {
    return false;
  }
  const auto source_shape = getNodeColumnRefs(source);
  if (source_shape.size() != exprs_.size()) {
    return false;
  }
  for (size_t i = 0; i < exprs_.size(); ++i) {
    const auto col_ref = dynamic_cast<const ColumnRef*>(exprs_[i].get());
    CHECK(col_ref);
    CHECK_EQ(source, col_ref->node());
    // We should add the additional check that input->index() !=
    // source_shape[i].index(), but Calcite doesn't generate the right
    // Sort-Project-Sort sequence when joins are involved.
    if (col_ref->node() !=
        dynamic_cast<const ColumnRef*>(source_shape[i].get())->node()) {
      return false;
    }
  }
  return true;
}

bool Project::isRenaming() const {
  if (!isSimple()) {
    return false;
  }
  CHECK_EQ(exprs_.size(), fields_.size());
  for (size_t i = 0; i < fields_.size(); ++i) {
    auto col_ref = dynamic_cast<const ColumnRef*>(exprs_[i].get());
    CHECK(col_ref);
    if (isRenamedInput(col_ref->node(), col_ref->index(), fields_[i])) {
      return true;
    }
  }
  return false;
}

Aggregate::Aggregate(Aggregate const& rhs)
    : Node(rhs)
    , groupby_count_(rhs.groupby_count_)
    , aggs_(rhs.aggs_)
    , fields_(rhs.fields_) {}

void Aggregate::rewriteExprs(hdk::ir::ExprRewriter& rewriter) {
  for (size_t i = 0; i < aggs_.size(); ++i) {
    aggs_[i] = rewriter.visit(aggs_[i].get());
  }
}

Join::Join(Join const& rhs)
    : Node(rhs), condition_(rhs.condition_), join_type_(rhs.join_type_) {}

void Join::rewriteExprs(hdk::ir::ExprRewriter& rewriter) {
  if (condition_) {
    condition_ = rewriter.visit(condition_.get());
  }
}

Filter::Filter(Filter const& rhs) : Node(rhs), condition_(rhs.condition_) {}

void Filter::rewriteExprs(hdk::ir::ExprRewriter& rewriter) {
  condition_ = rewriter.visit(condition_.get());
}

bool Sort::hasEquivCollationOf(const Sort& that) const {
  if (collation_.size() != that.collation_.size()) {
    return false;
  }

  for (size_t i = 0, e = collation_.size(); i < e; ++i) {
    auto this_sort_key = collation_[i];
    auto that_sort_key = that.collation_[i];
    if (this_sort_key.getSortDir() != that_sort_key.getSortDir()) {
      return false;
    }
    if (this_sort_key.getNullsPosition() != that_sort_key.getNullsPosition()) {
      return false;
    }
    auto this_equiv_keys = getEquivCols(this, this_sort_key.getField());
    auto that_equiv_keys = getEquivCols(&that, that_sort_key.getField());
    std::vector<std::pair<const Node*, int>> intersect;
    std::set_intersection(this_equiv_keys.begin(),
                          this_equiv_keys.end(),
                          that_equiv_keys.begin(),
                          that_equiv_keys.end(),
                          std::back_inserter(intersect));
    if (intersect.empty()) {
      return false;
    }
  }
  return true;
}

LogicalValues::LogicalValues(LogicalValues const& rhs)
    : Node(rhs), tuple_type_(rhs.tuple_type_), values_(rhs.values_) {}

LogicalUnion::LogicalUnion(NodeInputs inputs, bool is_all)
    : Node(std::move(inputs)), is_all_(is_all) {
  CHECK_EQ(2u, inputs_.size());
  if (!is_all_) {
    throw QueryNotSupported("UNION without ALL is not supported yet.");
  }
}

size_t LogicalUnion::size() const {
  return inputs_.front()->size();
}

std::string LogicalUnion::toString() const {
  return cat(::typeName(this), "(is_all(", is_all_, "))");
}

size_t LogicalUnion::toHash() const {
  if (!hash_) {
    hash_ = typeid(LogicalUnion).hash_code();
    boost::hash_combine(*hash_, is_all_);
  }
  return *hash_;
}

std::string LogicalUnion::getFieldName(const size_t i) const {
  if (auto const* input = dynamic_cast<Project const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<LogicalUnion const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<Aggregate const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<Scan const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  }
  UNREACHABLE() << "Unhandled input type: " << ::toString(inputs_.front());
  return {};
}

void LogicalUnion::checkForMatchingMetaInfoTypes() const {
  std::vector<TargetMetaInfo> const& tmis0 = inputs_[0]->getOutputMetainfo();
  std::vector<TargetMetaInfo> const& tmis1 = inputs_[1]->getOutputMetainfo();
  if (tmis0.size() != tmis1.size()) {
    VLOG(2) << "tmis0.size() = " << tmis0.size() << " != " << tmis1.size()
            << " = tmis1.size()";
    throw std::runtime_error("Subqueries of a UNION must have matching data types.");
  }
  for (size_t i = 0; i < tmis0.size(); ++i) {
    if (!tmis0[i].type()->equal(tmis1[i].type())) {
      auto ti0 = tmis0[i].type();
      auto ti1 = tmis1[i].type();
      VLOG(2) << "Types do not match for UNION:\n  tmis0[" << i
              << "].type()->toString() = " << ti0->toString() << "\n  tmis1[" << i
              << "].type()->toString() = " << ti1->toString();
      if (ti0->isExtDictionary() && ti1->isExtDictionary() &&
          (ti0->as<ExtDictionaryType>()->dictId() !=
           ti1->as<ExtDictionaryType>()->dictId())) {
        throw std::runtime_error(
            "Taking the UNION of different text-encoded dictionaries is not yet "
            "supported. This may be resolved by using shared dictionaries. For example, "
            "by making one a shared dictionary reference to the other.");
      } else {
        throw std::runtime_error(
            "Subqueries of a UNION must have the exact same data types.");
      }
    }
  }
}

namespace {

void collectNodes(NodePtr node, std::vector<NodePtr> nodes) {
  for (size_t i = 0; i < node->inputCount(); ++i) {
    collectNodes(std::const_pointer_cast<Node>(node->getAndOwnInput(i)), nodes);
  }
  nodes.push_back(node);
}

}  // namespace

QueryDag::QueryDag(ConfigPtr config, NodePtr root) : config_(config), root_(root) {
  collectNodes(root_, nodes_);
}

void QueryDag::eachNode(std::function<void(Node const*)> const& callback) const {
  for (auto const& node : nodes_) {
    if (node) {
      callback(node.get());
    }
  }
}

void QueryDag::resetQueryExecutionState() {
  for (auto& node : nodes_) {
    if (node) {
      node->resetQueryExecutionState();
    }
  }
}

// TODO: always simply use node->size()
size_t getNodeColumnCount(const Node* node) {
  // Nodes that don't depend on input.
  if (is_one_of<Scan, Project, Aggregate, LogicalUnion, LogicalValues>(node)) {
    return node->size();
  }

  // Nodes that preserve size.
  if (is_one_of<Filter, Sort>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    return getNodeColumnCount(node->getInput(0));
  }

  // Join concatenates the outputs from the inputs.
  if (is_one_of<Join>(node)) {
    CHECK_EQ(size_t(2), node->inputCount());
    return getNodeColumnCount(node->getInput(0)) + getNodeColumnCount(node->getInput(1));
  }

  LOG(FATAL) << "Unhandled ra_node type: " << ::toString(node);
  return 0;
}

ExprPtrVector genColumnRefs(const Node* node, size_t count) {
  ExprPtrVector res;
  res.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    res.emplace_back(makeExpr<ColumnRef>(getColumnType(node, i), node, i));
  }
  return res;
}

ExprPtrVector getNodeColumnRefs(const Node* node) {
  // Nodes that don't depend on input.
  if (is_one_of<Scan,
                Project,
                Aggregate,
                LogicalUnion,
                LogicalValues,
                Filter,
                Sort,
                Join>(node)) {
    return genColumnRefs(node, getNodeColumnCount(node));
  }

  LOG(FATAL) << "Unhandled ra_node type: " << ::toString(node);
  return {};
}

ExprPtr getNodeColumnRef(const Node* node, unsigned index) {
  CHECK_LT(index, node->size());

  if (is_one_of<Scan,
                Project,
                Aggregate,
                LogicalUnion,
                LogicalValues,
                Filter,
                Sort,
                Join>(node)) {
    return makeExpr<ColumnRef>(getColumnType(node, index), node, index);
  }

  LOG(FATAL) << "Unhandled node type: " << ::toString(node);
  return nullptr;
}

ExprPtr getJoinInputColumnRef(const ColumnRef* col_ref) {
  auto* node = col_ref->node();
  CHECK(node->is<Join>());
  CHECK(col_ref->index() < node->size());
  return col_ref->index() < node->getInput(0)->size()
             ? getNodeColumnRef(node->getInput(0), col_ref->index())
             : getNodeColumnRef(node->getInput(1),
                                col_ref->index() - node->getInput(0)->size());
}

const Type* getColumnType(const Node* node, size_t col_idx) {
  // By default use metainfo.
  const auto& metainfo = node->getOutputMetainfo();
  if (metainfo.size() > col_idx) {
    return metainfo[col_idx].type();
  }

  // For scans we can use embedded column info.
  const auto scan = dynamic_cast<const Scan*>(node);
  if (scan) {
    return scan->getColumnType(col_idx);
  }

  // For filter, sort and union we can propagate column type of
  // their sources.
  if (is_one_of<Filter, Sort, LogicalUnion>(node)) {
    return getColumnType(node->getInput(0), col_idx);
  }

  // For aggregates we can we can propagate type from group key
  // or extract type from AggExpr
  const auto agg = dynamic_cast<const Aggregate*>(node);
  if (agg) {
    if (col_idx < agg->getGroupByCount()) {
      return getColumnType(agg->getInput(0), col_idx);
    } else {
      return agg->getAggs()[col_idx - agg->getGroupByCount()]->type();
    }
  }

  // For logical values we can use its tuple type.
  const auto values = dynamic_cast<const LogicalValues*>(node);
  if (values) {
    CHECK_GT(values->size(), col_idx);
    auto type = values->getTupleType()[col_idx].type();
    if (type->isNull()) {
      // replace w/ bigint
      return type->ctx().int64();
    }
    return values->getTupleType()[col_idx].type();
  }

  // For projections type can be extracted from Exprs.
  const auto proj = dynamic_cast<const Project*>(node);
  if (proj) {
    CHECK_GT(proj->getExprs().size(), col_idx);
    return proj->getExprs()[col_idx]->type();
  }

  // For joins we can propagate type from one of its sources.
  const auto join = dynamic_cast<const Join*>(node);
  if (join) {
    CHECK_GT(join->size(), col_idx);
    if (col_idx < join->getInput(0)->size()) {
      return getColumnType(join->getInput(0), col_idx);
    } else {
      return getColumnType(join->getInput(1), col_idx - join->getInput(0)->size());
    }
  }

  CHECK(false) << "Missing output metainfo for node " + node->toString() +
                      " col_idx=" + std::to_string(col_idx);
  return {};
}

}  // namespace hdk::ir
