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

#include "RelAlgOptimizer.h"
#include "IR/ExprCollector.h"
#include "IR/ExprRewriter.h"
#include "Logger/Logger.h"
#include "Visitors/SubQueryCollector.h"

#include <boost/make_unique.hpp>
#include <numeric>
#include <string>
#include <unordered_map>

extern size_t g_max_log_length;

namespace {

class ProjectInputRedirector : public hdk::ir::ExprRewriter {
 public:
  ProjectInputRedirector(const std::unordered_set<const hdk::ir::Project*>& crt_inputs)
      : crt_projects_(crt_inputs) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    auto source = dynamic_cast<const hdk::ir::Project*>(col_ref->node());
    if (source && crt_projects_.count(source)) {
      auto new_source = source->getInput(0);
      auto new_col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(
          source->getExpr(col_ref->index()).get());
      if (new_col_ref) {
        CHECK_EQ(new_col_ref->node(), new_source);
        return new_col_ref->shared();
      }
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

 private:
  const std::unordered_set<const hdk::ir::Project*>& crt_projects_;
};

// TODO: use more generic InputRenumberVisitor instead.
template <bool bAllowMissing>
class InputSimpleRenumberVisitor : public hdk::ir::ExprRewriter {
 public:
  InputSimpleRenumberVisitor(const std::unordered_map<size_t, size_t>& new_numbering)
      : old_to_new_idx_(new_numbering) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    auto renum_it = old_to_new_idx_.find(col_ref->index());
    if constexpr (bAllowMissing) {
      if (renum_it != old_to_new_idx_.end()) {
        return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            col_ref->type(), col_ref->node(), renum_it->second);
      } else {
        return ExprRewriter::visitColumnRef(col_ref);
      }
    } else {
      CHECK(renum_it != old_to_new_idx_.end());
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          col_ref->type(), col_ref->node(), renum_it->second);
    }
  }

 private:
  const std::unordered_map<size_t, size_t>& old_to_new_idx_;
};

class RebindInputsVisitor : public hdk::ir::ExprRewriter {
 public:
  RebindInputsVisitor(const hdk::ir::Node* old_input, const hdk::ir::Node* new_input)
      : old_input_(old_input), new_input_(new_input) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (col_ref->node() == old_input_) {
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          col_ref->type(), new_input_, col_ref->index());
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

  void visitNode(const hdk::ir::Node* node) {
    if (dynamic_cast<const hdk::ir::Aggregate*>(node) ||
        dynamic_cast<const hdk::ir::Sort*>(node)) {
      return;
    }
    if (auto join =
            const_cast<hdk::ir::Join*>(dynamic_cast<const hdk::ir::Join*>(node))) {
      if (join->getCondition()) {
        auto cond = ExprRewriter::visit(join->getCondition());
        join->setCondition(std::move(cond));
      }
      return;
    }
    if (auto project =
            const_cast<hdk::ir::Project*>(dynamic_cast<const hdk::ir::Project*>(node))) {
      hdk::ir::ExprPtrVector new_exprs;
      for (auto& expr : project->getExprs()) {
        new_exprs.push_back(ExprRewriter::visit(expr.get()));
      }
      project->setExpressions(std::move(new_exprs));
      return;
    }
    if (auto filter =
            const_cast<hdk::ir::Filter*>(dynamic_cast<const hdk::ir::Filter*>(node))) {
      auto cond = ExprRewriter::visit(filter->getConditionExpr());
      filter->setCondition(std::move(cond));
      return;
    }
    CHECK(false);
  }

 private:
  const hdk::ir::Node* old_input_;
  const hdk::ir::Node* new_input_;
};

size_t get_actual_source_size(
    const hdk::ir::Project* curr_project,
    const std::unordered_set<const hdk::ir::Project*>& projects_to_remove) {
  auto source = curr_project->getInput(0);
  while (auto filter = dynamic_cast<const hdk::ir::Filter*>(source)) {
    source = filter->getInput(0);
  }
  if (auto src_project = dynamic_cast<const hdk::ir::Project*>(source)) {
    if (projects_to_remove.count(src_project)) {
      return get_actual_source_size(src_project, projects_to_remove);
    }
  }
  return curr_project->getInput(0)->size();
}

bool safe_to_redirect(
    const hdk::ir::Project* project,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web) {
  if (!project->isSimple()) {
    return false;
  }
  auto usrs_it = du_web.find(project);
  CHECK(usrs_it != du_web.end());
  for (auto usr : usrs_it->second) {
    if (!dynamic_cast<const hdk::ir::Project*>(usr) &&
        !dynamic_cast<const hdk::ir::Filter*>(usr)) {
      return false;
    }
  }
  return true;
}

bool is_identical_copy(
    const hdk::ir::Project* project,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web,
    const std::unordered_set<const hdk::ir::Project*>& projects_to_remove,
    std::unordered_set<const hdk::ir::Project*>& permutating_projects) {
  auto source_size = get_actual_source_size(project, projects_to_remove);
  if (project->size() > source_size) {
    return false;
  }

  if (project->size() < source_size) {
    auto usrs_it = du_web.find(project);
    CHECK(usrs_it != du_web.end());
    bool guard_found = false;
    while (usrs_it->second.size() == size_t(1)) {
      auto only_usr = *usrs_it->second.begin();
      if (dynamic_cast<const hdk::ir::Project*>(only_usr)) {
        guard_found = true;
        break;
      }
      if (dynamic_cast<const hdk::ir::Aggregate*>(only_usr) ||
          dynamic_cast<const hdk::ir::Sort*>(only_usr) ||
          dynamic_cast<const hdk::ir::Join*>(only_usr) ||
          dynamic_cast<const hdk::ir::LogicalUnion*>(only_usr)) {
        return false;
      }
      CHECK(dynamic_cast<const hdk::ir::Filter*>(only_usr))
          << "only_usr: " << only_usr->toString();
      usrs_it = du_web.find(only_usr);
      CHECK(usrs_it != du_web.end());
    }

    if (!guard_found) {
      return false;
    }
  }

  bool identical = true;
  for (size_t i = 0; i < project->size(); ++i) {
    auto target = dynamic_cast<const hdk::ir::ColumnRef*>(project->getExpr(i).get());
    CHECK(target);
    if (i != target->index()) {
      identical = false;
      break;
    }
  }

  if (identical) {
    return true;
  }

  if (safe_to_redirect(project, du_web)) {
    permutating_projects.insert(project);
    return true;
  }

  return false;
}

void propagate_rex_input_renumber(
    const hdk::ir::Filter* excluded_root,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web) {
  CHECK(excluded_root);
  auto src_project = dynamic_cast<const hdk::ir::Project*>(excluded_root->getInput(0));
  CHECK(src_project && src_project->isSimple());
  const auto indirect_join_src =
      dynamic_cast<const hdk::ir::Join*>(src_project->getInput(0));
  std::unordered_map<size_t, size_t> old_to_new_idx;
  for (size_t i = 0; i < src_project->size(); ++i) {
    auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(src_project->getExpr(i).get());
    CHECK(col_ref);
    size_t src_base = 0;
    if (indirect_join_src != nullptr &&
        indirect_join_src->getInput(1) == col_ref->node()) {
      src_base = indirect_join_src->getInput(0)->size();
    }
    old_to_new_idx.insert(std::make_pair(i, src_base + col_ref->index()));
    old_to_new_idx.insert(std::make_pair(i, col_ref->index()));
  }
  CHECK(old_to_new_idx.size());
  InputSimpleRenumberVisitor<false> visitor(old_to_new_idx);
  auto usrs_it = du_web.find(excluded_root);
  CHECK(usrs_it != du_web.end());
  std::vector<const hdk::ir::Node*> work_set(usrs_it->second.begin(),
                                             usrs_it->second.end());
  while (!work_set.empty()) {
    auto node = work_set.back();
    work_set.pop_back();
    auto modified_node = const_cast<hdk::ir::Node*>(node);
    if (auto filter = dynamic_cast<hdk::ir::Filter*>(modified_node)) {
      auto new_condition_expr = visitor.visit(filter->getConditionExpr());
      filter->setCondition(std::move(new_condition_expr));
      auto usrs_it = du_web.find(filter);
      CHECK(usrs_it != du_web.end() && usrs_it->second.size() == 1);
      work_set.push_back(*usrs_it->second.begin());
      continue;
    }
    if (auto project = dynamic_cast<hdk::ir::Project*>(modified_node)) {
      hdk::ir::ExprPtrVector new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(visitor.visit(project->getExpr(i).get()));
      }
      project->setExpressions(std::move(new_exprs));
      continue;
    }
    CHECK(false);
  }
}

// This function appears to redirect/remove redundant Projection input nodes(?)
void redirect_inputs_of(
    std::shared_ptr<hdk::ir::Node> node,
    const std::unordered_set<const hdk::ir::Project*>& projects,
    const std::unordered_set<const hdk::ir::Project*>& permutating_projects,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web) {
  if (dynamic_cast<hdk::ir::LogicalUnion*>(node.get())) {
    return;  // UNION keeps all Projection inputs.
  }
  std::shared_ptr<const hdk::ir::Project> src_project = nullptr;
  for (size_t i = 0; i < node->inputCount(); ++i) {
    if (auto project =
            std::dynamic_pointer_cast<const hdk::ir::Project>(node->getAndOwnInput(i))) {
      if (projects.count(project.get())) {
        src_project = project;
        break;
      }
    }
  }
  if (!src_project) {
    return;
  }
  if (auto join = std::dynamic_pointer_cast<hdk::ir::Join>(node)) {
    auto other_project =
        src_project == node->getAndOwnInput(0)
            ? std::dynamic_pointer_cast<const hdk::ir::Project>(node->getAndOwnInput(1))
            : std::dynamic_pointer_cast<const hdk::ir::Project>(node->getAndOwnInput(0));
    join->replaceInput(src_project, src_project->getAndOwnInput(0));
    RebindInputsVisitor rebinder(src_project.get(), src_project->getInput(0));
    auto usrs_it = du_web.find(join.get());
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      rebinder.visitNode(usr);
    }

    if (other_project && projects.count(other_project.get())) {
      join->replaceInput(other_project, other_project->getAndOwnInput(0));
      RebindInputsVisitor other_rebinder(other_project.get(), other_project->getInput(0));
      for (auto usr : usrs_it->second) {
        other_rebinder.visitNode(usr);
      }
    }
    return;
  }
  if (auto project = std::dynamic_pointer_cast<hdk::ir::Project>(node)) {
    project->hdk::ir::Node::replaceInput(src_project, src_project->getAndOwnInput(0));
    ProjectInputRedirector visitor(projects);
    hdk::ir::ExprPtrVector new_exprs;
    for (auto& expr : project->getExprs()) {
      new_exprs.push_back(visitor.visit(expr.get()));
    }
    project->setExpressions(std::move(new_exprs));
    return;
  }
  if (auto filter = std::dynamic_pointer_cast<hdk::ir::Filter>(node)) {
    const bool is_permutating_proj = permutating_projects.count(src_project.get());
    if (is_permutating_proj ||
        dynamic_cast<const hdk::ir::Join*>(src_project->getInput(0))) {
      if (is_permutating_proj) {
        propagate_rex_input_renumber(filter.get(), du_web);
      }
      filter->hdk::ir::Node::replaceInput(src_project, src_project->getAndOwnInput(0));
      ProjectInputRedirector visitor(projects);
      auto new_condition_expr = visitor.visit(filter->getConditionExpr());
      filter->setCondition(new_condition_expr);
    } else {
      filter->replaceInput(src_project, src_project->getAndOwnInput(0));
    }
    return;
  }
  if (std::dynamic_pointer_cast<hdk::ir::Sort>(node)) {
    auto const src_project_input = src_project->getInput(0);
    if (dynamic_cast<const hdk::ir::Scan*>(src_project_input) ||
        dynamic_cast<const hdk::ir::LogicalValues*>(src_project_input) ||
        dynamic_cast<const hdk::ir::LogicalUnion*>(src_project_input)) {
      return;
    }
  }
  CHECK(std::dynamic_pointer_cast<hdk::ir::Aggregate>(node) ||
        std::dynamic_pointer_cast<hdk::ir::Sort>(node));
  node->replaceInput(src_project, src_project->getAndOwnInput(0));
}

void cleanup_dead_nodes(std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) {
  for (auto nodeIt = nodes.rbegin(); nodeIt != nodes.rend(); ++nodeIt) {
    if (nodeIt->use_count() == 1) {
      VLOG(1) << "Node (ID: " << (*nodeIt)->getId() << ") deleted.";
      if (logger::fast_logging_check(logger::Severity::DEBUG2)) {
        auto node_str = (*nodeIt)->toString();
        auto [node_substr, post_fix] = ::substring(node_str, g_max_log_length);
        VLOG(2) << "Deleted Node (ID: " << (*nodeIt)->getId()
                << ") contents: " << node_substr << post_fix;
      }
      nodeIt->reset();
    }
  }

  std::vector<std::shared_ptr<hdk::ir::Node>> new_nodes;
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    new_nodes.push_back(node);
  }
  nodes.swap(new_nodes);
}

std::unordered_set<const hdk::ir::Project*> get_visible_projects(
    const hdk::ir::Node* root) {
  if (auto project = dynamic_cast<const hdk::ir::Project*>(root)) {
    return {project};
  }

  if (dynamic_cast<const hdk::ir::Aggregate*>(root) ||
      dynamic_cast<const hdk::ir::Scan*>(root) ||
      dynamic_cast<const hdk::ir::LogicalValues*>(root)) {
    return std::unordered_set<const hdk::ir::Project*>{};
  }

  if (auto join = dynamic_cast<const hdk::ir::Join*>(root)) {
    auto lhs_projs = get_visible_projects(join->getInput(0));
    auto rhs_projs = get_visible_projects(join->getInput(1));
    lhs_projs.insert(rhs_projs.begin(), rhs_projs.end());
    return lhs_projs;
  }

  if (auto logical_union = dynamic_cast<const hdk::ir::LogicalUnion*>(root)) {
    auto projections = get_visible_projects(logical_union->getInput(0));
    for (size_t i = 1; i < logical_union->inputCount(); ++i) {
      auto next = get_visible_projects(logical_union->getInput(i));
      projections.insert(next.begin(), next.end());
    }
    return projections;
  }

  CHECK(dynamic_cast<const hdk::ir::Filter*>(root) ||
        dynamic_cast<const hdk::ir::Sort*>(root))
      << "root = " << root->toString();
  return get_visible_projects(root->getInput(0));
}

// TODO(miyu): checking this at runtime is more accurate
bool is_distinct(const size_t input_idx, const hdk::ir::Node* node) {
  if (dynamic_cast<const hdk::ir::Filter*>(node) ||
      dynamic_cast<const hdk::ir::Sort*>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    return is_distinct(input_idx, node->getInput(0));
  }
  if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    if (aggregate->getGroupByCount() == 1 && !input_idx) {
      return true;
    }
    if (input_idx < aggregate->getGroupByCount()) {
      return is_distinct(input_idx, node->getInput(0));
    }
    return false;
  }
  if (auto project = dynamic_cast<const hdk::ir::Project*>(node)) {
    CHECK_LT(input_idx, project->size());
    if (auto input =
            dynamic_cast<const hdk::ir::ColumnRef*>(project->getExpr(input_idx).get())) {
      CHECK_EQ(size_t(1), node->inputCount());
      return is_distinct(input->index(), project->getInput(0));
    }
    return false;
  }
  CHECK(dynamic_cast<const hdk::ir::Join*>(node) ||
        dynamic_cast<const hdk::ir::Scan*>(node));
  return false;
}

}  // namespace

std::unordered_map<const hdk::ir::Node*, std::unordered_set<const hdk::ir::Node*>>
build_du_web(const std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  std::unordered_map<const hdk::ir::Node*, std::unordered_set<const hdk::ir::Node*>> web;
  std::unordered_set<const hdk::ir::Node*> visited;
  std::vector<const hdk::ir::Node*> work_set;
  for (auto node : nodes) {
    if (std::dynamic_pointer_cast<hdk::ir::Scan>(node) || visited.count(node.get())) {
      continue;
    }
    work_set.push_back(node.get());
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      if (visited.count(walker)) {
        continue;
      }
      CHECK(!web.count(walker));
      auto it_ok =
          web.insert(std::make_pair(walker, std::unordered_set<const hdk::ir::Node*>{}));
      CHECK(it_ok.second);
      visited.insert(walker);
      CHECK(dynamic_cast<const hdk::ir::Join*>(walker) ||
            dynamic_cast<const hdk::ir::Project*>(walker) ||
            dynamic_cast<const hdk::ir::Aggregate*>(walker) ||
            dynamic_cast<const hdk::ir::Filter*>(walker) ||
            dynamic_cast<const hdk::ir::Sort*>(walker) ||
            dynamic_cast<const hdk::ir::LogicalValues*>(walker) ||
            dynamic_cast<const hdk::ir::LogicalUnion*>(walker));
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(i);
        if (dynamic_cast<const hdk::ir::Scan*>(src)) {
          continue;
        }
        if (web.empty() || !web.count(src)) {
          web.insert(std::make_pair(src, std::unordered_set<const hdk::ir::Node*>{}));
        }
        web[src].insert(walker);
        work_set.push_back(src);
      }
    }
  }
  return web;
}

/**
 * Return true if the input project separates two sort nodes, i.e. Sort -> Project ->
 * Sort. This pattern often occurs in machine generated SQL, e.g. SELECT * FROM (SELECT *
 * FROM t LIMIT 10) t0 LIMIT 1;
 * Use this function to prevent optimizing out the intermediate project, as the project is
 * required to ensure the first sort runs to completion prior to the second sort. Back to
 * back sort nodes are not executable and will throw an error.
 */
bool project_separates_sort(const hdk::ir::Project* project,
                            const hdk::ir::Node* next_node) {
  CHECK(project);
  if (!next_node) {
    return false;
  }

  auto sort = dynamic_cast<const hdk::ir::Sort*>(next_node);
  if (!sort) {
    return false;
  }
  if (!(project->inputCount() == 1)) {
    return false;
  }

  if (dynamic_cast<const hdk::ir::Sort*>(project->getInput(0))) {
    return true;
  }
  return false;
}

// For now, the only target to eliminate is restricted to project-aggregate pair between
// scan/sort and join
// TODO(miyu): allow more chance if proved safe
void eliminate_identical_copy(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  std::unordered_set<std::shared_ptr<const hdk::ir::Node>> copies;
  auto sink = nodes.back();
  for (auto node : nodes) {
    auto aggregate = std::dynamic_pointer_cast<const hdk::ir::Aggregate>(node);
    if (!aggregate || aggregate == sink ||
        !(aggregate->getGroupByCount() == 1 && aggregate->getAggsCount() == 0)) {
      continue;
    }
    auto project =
        std::dynamic_pointer_cast<const hdk::ir::Project>(aggregate->getAndOwnInput(0));
    if (project && project->size() == aggregate->size() &&
        project->getFields() == aggregate->getFields()) {
      CHECK_EQ(size_t(0), copies.count(aggregate));
      copies.insert(aggregate);
    }
  }
  for (auto node : nodes) {
    if (!node->inputCount()) {
      continue;
    }
    auto last_source = node->getAndOwnInput(node->inputCount() - 1);
    if (!copies.count(last_source)) {
      continue;
    }
    auto aggregate = std::dynamic_pointer_cast<const hdk::ir::Aggregate>(last_source);
    CHECK(aggregate);
    if (!std::dynamic_pointer_cast<const hdk::ir::Join>(node) || aggregate->size() != 1) {
      continue;
    }
    auto project =
        std::dynamic_pointer_cast<const hdk::ir::Project>(aggregate->getAndOwnInput(0));
    CHECK(project);
    CHECK_EQ(size_t(1), project->size());
    if (!is_distinct(size_t(0), project.get())) {
      continue;
    }
    auto new_source = project->getAndOwnInput(0);
    if (std::dynamic_pointer_cast<const hdk::ir::Sort>(new_source) ||
        std::dynamic_pointer_cast<const hdk::ir::Scan>(new_source)) {
      node->replaceInput(last_source, new_source);
    }
  }
  decltype(copies)().swap(copies);

  auto web = build_du_web(nodes);

  std::unordered_set<const hdk::ir::Project*> projects;
  std::unordered_set<const hdk::ir::Project*> permutating_projects;
  auto const visible_projs = get_visible_projects(nodes.back().get());
  for (auto node_it = nodes.begin(); node_it != nodes.end(); node_it++) {
    auto node = *node_it;
    auto project = std::dynamic_pointer_cast<hdk::ir::Project>(node);
    auto next_node_it = std::next(node_it);
    if (project && project->isSimple() &&
        (!visible_projs.count(project.get()) || !project->isRenaming()) &&
        is_identical_copy(project.get(), web, projects, permutating_projects) &&
        !project_separates_sort(
            project.get(), next_node_it == nodes.end() ? nullptr : next_node_it->get())) {
      projects.insert(project.get());
    }
  }

  for (auto node : nodes) {
    redirect_inputs_of(node, projects, permutating_projects, web);
  }

  cleanup_dead_nodes(nodes);
}

namespace {

using InputSet =
    std::unordered_set<std::pair<const hdk::ir::Node*, unsigned>,
                       boost::hash<std::pair<const hdk::ir::Node*, unsigned>>>;

class InputCollector : public hdk::ir::ExprCollector<InputSet, InputCollector> {
 private:
  const hdk::ir::Node* node_;

 public:
  InputCollector(const hdk::ir::Node* node) : node_(node) {}

  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (node_->inputCount() == 1) {
      auto src = node_->getInput(0);
      if (auto join = dynamic_cast<const hdk::ir::Join*>(src)) {
        CHECK_EQ(join->inputCount(), size_t(2));
        const auto src2_in_offset = join->getInput(0)->size();
        if (col_ref->node() == join->getInput(1)) {
          result_.emplace(src, col_ref->index() + src2_in_offset);
        } else {
          result_.emplace(src, col_ref->index());
        }
        return;
      }
    }
    result_.emplace(col_ref->node(), col_ref->index());
  }
};

size_t pick_always_live_col_idx(const hdk::ir::Node* node) {
  CHECK(node->size());
  if (auto filter = dynamic_cast<const hdk::ir::Filter*>(node)) {
    auto rex_ins = InputCollector::collect(filter->getConditionExpr(), node);
    if (!rex_ins.empty()) {
      return static_cast<size_t>(rex_ins.begin()->second);
    }
    return pick_always_live_col_idx(filter->getInput(0));
  } else if (auto join = dynamic_cast<const hdk::ir::Join*>(node)) {
    auto inputs = InputCollector::collect(join->getCondition(), node);
    if (!inputs.empty()) {
      return static_cast<size_t>(inputs.begin()->second);
    }
    if (auto lhs_idx = pick_always_live_col_idx(join->getInput(0))) {
      return lhs_idx;
    }
    if (auto rhs_idx = pick_always_live_col_idx(join->getInput(0))) {
      return rhs_idx + join->getInput(0)->size();
    }
  } else if (auto sort = dynamic_cast<const hdk::ir::Sort*>(node)) {
    if (sort->collationCount()) {
      return sort->getCollation(0).getField();
    }
    return pick_always_live_col_idx(sort->getInput(0));
  }
  return size_t(0);
}

std::vector<std::unordered_set<size_t>> get_live_ins(
    const hdk::ir::Node* node,
    const std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>>&
        live_outs) {
  if (!node || dynamic_cast<const hdk::ir::Scan*>(node)) {
    return {};
  }
  auto it = live_outs.find(node);
  CHECK(it != live_outs.end());
  auto live_out = it->second;
  if (auto project = dynamic_cast<const hdk::ir::Project*>(node)) {
    CHECK_EQ(size_t(1), project->inputCount());
    std::unordered_set<size_t> live_in;
    for (const auto& idx : live_out) {
      CHECK_LT(idx, project->size());
      auto partial_in = InputCollector::collect(project->getExpr(idx).get(), node);
      for (auto rex_in : partial_in) {
        live_in.insert(rex_in.second);
      }
    }
    if (project->size() == 1 &&
        dynamic_cast<const hdk::ir::Constant*>(project->getExpr(0).get())) {
      CHECK(live_in.empty());
      live_in.insert(pick_always_live_col_idx(project->getInput(0)));
    }
    return {live_in};
  }
  if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
    CHECK_EQ(size_t(1), aggregate->inputCount());
    const auto group_key_count = static_cast<size_t>(aggregate->getGroupByCount());
    const auto agg_expr_count = static_cast<size_t>(aggregate->getAggsCount());
    std::unordered_set<size_t> live_in;
    for (size_t i = 0; i < group_key_count; ++i) {
      live_in.insert(i);
    }
    bool has_count_star_only{false};
    for (const auto& idx : live_out) {
      if (idx < group_key_count) {
        continue;
      }
      const auto agg_idx = idx - group_key_count;
      CHECK_LT(agg_idx, agg_expr_count);
      auto agg_expr =
          dynamic_cast<const hdk::ir::AggExpr*>(aggregate->getAgg(agg_idx).get());
      CHECK(agg_expr);

      if (!agg_expr->arg()) {
        has_count_star_only = true;
      } else {
        auto inputs = InputCollector::collect(agg_expr, node);
        for (auto& pr : inputs) {
          CHECK_EQ(aggregate->getInput(0), pr.first);
          live_in.insert(pr.second);
        }
      }
    }
    if (has_count_star_only && !group_key_count) {
      live_in.insert(size_t(0));
    }
    return {live_in};
  }
  if (auto join = dynamic_cast<const hdk::ir::Join*>(node)) {
    std::unordered_set<size_t> lhs_live_ins;
    std::unordered_set<size_t> rhs_live_ins;
    CHECK_EQ(size_t(2), join->inputCount());
    auto lhs = join->getInput(0);
    auto rhs = join->getInput(1);
    const auto rhs_idx_base = lhs->size();
    for (const auto idx : live_out) {
      if (idx < rhs_idx_base) {
        lhs_live_ins.insert(idx);
      } else {
        rhs_live_ins.insert(idx - rhs_idx_base);
      }
    }
    auto inputs = InputCollector::collect(join->getCondition(), node);
    for (const auto& input : inputs) {
      const auto in_idx = static_cast<size_t>(input.second);
      if (input.first == lhs) {
        lhs_live_ins.insert(in_idx);
        continue;
      }
      if (input.first == rhs) {
        rhs_live_ins.insert(in_idx);
        continue;
      }
      CHECK(false);
    }
    return {lhs_live_ins, rhs_live_ins};
  }
  if (auto sort = dynamic_cast<const hdk::ir::Sort*>(node)) {
    CHECK_EQ(size_t(1), sort->inputCount());
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    for (size_t i = 0; i < sort->collationCount(); ++i) {
      live_in.insert(sort->getCollation(i).getField());
    }
    return {live_in};
  }
  if (auto filter = dynamic_cast<const hdk::ir::Filter*>(node)) {
    CHECK_EQ(size_t(1), filter->inputCount());
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    auto rex_ins = InputCollector::collect(filter->getConditionExpr(), node);
    for (const auto& rex_in : rex_ins) {
      live_in.insert(static_cast<size_t>(rex_in.second));
    }
    return {live_in};
  }
  if (auto logical_union = dynamic_cast<const hdk::ir::LogicalUnion*>(node)) {
    return std::vector<std::unordered_set<size_t>>(logical_union->inputCount(), live_out);
  }
  return {};
}

bool any_dead_col_in(const hdk::ir::Node* node,
                     const std::unordered_set<size_t>& live_outs) {
  CHECK(!dynamic_cast<const hdk::ir::Scan*>(node));
  if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
    for (size_t i = aggregate->getGroupByCount(); i < aggregate->size(); ++i) {
      if (!live_outs.count(i)) {
        return true;
      }
    }
    return false;
  }

  return node->size() > live_outs.size();
}

bool does_redef_cols(const hdk::ir::Node* node) {
  return dynamic_cast<const hdk::ir::Aggregate*>(node) ||
         dynamic_cast<const hdk::ir::Project*>(node);
}

class AvailabilityChecker {
 public:
  AvailabilityChecker(
      const std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
          liveouts,
      const std::unordered_set<const hdk::ir::Node*>& intact_nodes)
      : liveouts_(liveouts), intact_nodes_(intact_nodes) {}

  bool hasAllSrcReady(const hdk::ir::Node* node) const {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      auto src = node->getInput(i);
      if (!dynamic_cast<const hdk::ir::Scan*>(src) &&
          liveouts_.find(src) == liveouts_.end() && !intact_nodes_.count(src)) {
        return false;
      }
    }
    return true;
  }

 private:
  const std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
      liveouts_;
  const std::unordered_set<const hdk::ir::Node*>& intact_nodes_;
};

void add_new_indices_for(
    const hdk::ir::Node* node,
    std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
        new_liveouts,
    const std::unordered_set<size_t>& old_liveouts,
    const std::unordered_set<const hdk::ir::Node*>& intact_nodes,
    const std::unordered_map<const hdk::ir::Node*, size_t>& orig_node_sizes) {
  auto live_fields = old_liveouts;
  if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
    for (size_t i = 0; i < aggregate->getGroupByCount(); ++i) {
      live_fields.insert(i);
    }
  }
  auto it_ok =
      new_liveouts.insert(std::make_pair(node, std::unordered_map<size_t, size_t>{}));
  CHECK(it_ok.second);
  auto& new_indices = it_ok.first->second;
  if (intact_nodes.count(node)) {
    for (size_t i = 0, e = node->size(); i < e; ++i) {
      new_indices.insert(std::make_pair(i, i));
    }
    return;
  }
  if (does_redef_cols(node)) {
    auto node_sz_it = orig_node_sizes.find(node);
    CHECK(node_sz_it != orig_node_sizes.end());
    const auto node_size = node_sz_it->second;
    CHECK_GT(node_size, live_fields.size());
    auto node_str = node->toString();
    auto [node_substr, post_fix] = ::substring(node_str, g_max_log_length);
    LOG(INFO) << node_substr << post_fix << " eliminated "
              << node_size - live_fields.size() << " columns.";
    std::vector<size_t> ordered_indices(live_fields.begin(), live_fields.end());
    std::sort(ordered_indices.begin(), ordered_indices.end());
    for (size_t i = 0; i < ordered_indices.size(); ++i) {
      new_indices.insert(std::make_pair(ordered_indices[i], i));
    }
    return;
  }
  std::vector<size_t> ordered_indices;
  for (size_t i = 0, old_base = 0, new_base = 0; i < node->inputCount(); ++i) {
    auto src = node->getInput(i);
    auto src_renum_it = new_liveouts.find(src);
    if (src_renum_it != new_liveouts.end()) {
      for (auto m : src_renum_it->second) {
        new_indices.insert(std::make_pair(old_base + m.first, new_base + m.second));
      }
      new_base += src_renum_it->second.size();
    } else if (dynamic_cast<const hdk::ir::Scan*>(src) || intact_nodes.count(src)) {
      for (size_t i = 0; i < src->size(); ++i) {
        new_indices.insert(std::make_pair(old_base + i, new_base + i));
      }
      new_base += src->size();
    } else {
      CHECK(false);
    }
    auto src_sz_it = orig_node_sizes.find(src);
    CHECK(src_sz_it != orig_node_sizes.end());
    old_base += src_sz_it->second;
  }
}

class InputRenumberVisitor : public hdk::ir::ExprRewriter {
 public:
  InputRenumberVisitor(
      const std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
          new_numbering)
      : node_to_input_renum_(new_numbering) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    auto node_it = node_to_input_renum_.find(col_ref->node());
    if (node_it != node_to_input_renum_.end()) {
      auto idx_it = node_it->second.find(col_ref->index());
      if (idx_it != node_it->second.end()) {
        return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            col_ref->type(), col_ref->node(), idx_it->second);
      }
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

 private:
  const std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
      node_to_input_renum_;
};

hdk::ir::SortField renumber_sort_field(
    const hdk::ir::SortField& old_field,
    const std::unordered_map<size_t, size_t>& new_numbering) {
  auto field_idx = old_field.getField();
  auto idx_it = new_numbering.find(field_idx);
  if (idx_it != new_numbering.end()) {
    field_idx = idx_it->second;
  }
  return hdk::ir::SortField(
      field_idx, old_field.getSortDir(), old_field.getNullsPosition());
}

std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>> mark_live_columns(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) {
  std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>> live_outs;
  std::vector<const hdk::ir::Node*> work_set;
  for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); ++node_it) {
    auto node = node_it->get();
    if (dynamic_cast<const hdk::ir::Scan*>(node) || live_outs.count(node)) {
      continue;
    }
    std::vector<size_t> all_live(node->size());
    std::iota(all_live.begin(), all_live.end(), size_t(0));
    live_outs.insert(std::make_pair(
        node, std::unordered_set<size_t>(all_live.begin(), all_live.end())));

    work_set.push_back(node);
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      CHECK(!dynamic_cast<const hdk::ir::Scan*>(walker));
      CHECK(live_outs.count(walker));
      auto live_ins = get_live_ins(walker, live_outs);
      CHECK_EQ(live_ins.size(), walker->inputCount());
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(i);
        if (dynamic_cast<const hdk::ir::Scan*>(src) || live_ins[i].empty()) {
          continue;
        }
        if (!live_outs.count(src)) {
          live_outs.insert(std::make_pair(src, std::unordered_set<size_t>{}));
        }
        auto src_it = live_outs.find(src);
        CHECK(src_it != live_outs.end());
        auto& live_out = src_it->second;
        bool changed = false;
        if (!live_out.empty()) {
          live_out.insert(live_ins[i].begin(), live_ins[i].end());
          changed = true;
        } else {
          for (int idx : live_ins[i]) {
            changed |= live_out.insert(idx).second;
          }
        }
        if (changed) {
          work_set.push_back(src);
        }
      }
    }
  }
  return live_outs;
}

std::string get_field_name(const hdk::ir::Node* node, size_t index) {
  CHECK_LT(index, node->size());
  if (auto scan = dynamic_cast<const hdk::ir::Scan*>(node)) {
    return scan->getFieldName(index);
  }
  if (auto aggregate = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
    CHECK_EQ(aggregate->size(), aggregate->getFields().size());
    return aggregate->getFieldName(index);
  }
  if (auto join = dynamic_cast<const hdk::ir::Join*>(node)) {
    const auto lhs_size = join->getInput(0)->size();
    if (index < lhs_size) {
      return get_field_name(join->getInput(0), index);
    }
    return get_field_name(join->getInput(1), index - lhs_size);
  }
  if (auto project = dynamic_cast<const hdk::ir::Project*>(node)) {
    return project->getFieldName(index);
  }
  CHECK(dynamic_cast<const hdk::ir::Sort*>(node) ||
        dynamic_cast<const hdk::ir::Filter*>(node));
  return get_field_name(node->getInput(0), index);
}

void try_insert_coalesceable_proj(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes,
    std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>>& liveouts,
    std::unordered_map<const hdk::ir::Node*, std::unordered_set<const hdk::ir::Node*>>&
        du_web) {
  std::vector<std::shared_ptr<hdk::ir::Node>> new_nodes;
  for (auto node : nodes) {
    new_nodes.push_back(node);
    if (!std::dynamic_pointer_cast<hdk::ir::Filter>(node)) {
      continue;
    }
    const auto filter = node.get();
    auto liveout_it = liveouts.find(filter);
    CHECK(liveout_it != liveouts.end());
    auto& outs = liveout_it->second;
    if (!any_dead_col_in(filter, outs)) {
      continue;
    }
    auto usrs_it = du_web.find(filter);
    CHECK(usrs_it != du_web.end());
    auto& usrs = usrs_it->second;
    if (usrs.size() != 1 || does_redef_cols(*usrs.begin())) {
      continue;
    }
    auto only_usr = const_cast<hdk::ir::Node*>(*usrs.begin());

    hdk::ir::ExprPtrVector exprs;
    std::vector<std::string> fields;
    for (size_t i = 0; i < filter->size(); ++i) {
      exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          getColumnType(filter, i), filter, static_cast<unsigned>(i)));
      fields.push_back(get_field_name(filter, i));
    }
    auto project_owner =
        std::make_shared<hdk::ir::Project>(std::move(exprs), fields, node);
    auto project = project_owner.get();

    only_usr->replaceInput(node, project_owner);
    if (dynamic_cast<const hdk::ir::Join*>(only_usr)) {
      RebindInputsVisitor visitor(filter, project);
      for (auto usr : du_web[only_usr]) {
        visitor.visitNode(usr);
      }
    }

    liveouts.insert(std::make_pair(project, outs));

    usrs.clear();
    usrs.insert(project);
    du_web.insert(
        std::make_pair(project, std::unordered_set<const hdk::ir::Node*>{only_usr}));

    new_nodes.push_back(project_owner);
  }
  if (new_nodes.size() > nodes.size()) {
    nodes.swap(new_nodes);
  }
}

std::pair<std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>,
          std::vector<const hdk::ir::Node*>>
sweep_dead_columns(
    const std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>>& live_outs,
    const std::vector<std::shared_ptr<hdk::ir::Node>>& nodes,
    const std::unordered_set<const hdk::ir::Node*>& intact_nodes,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web,
    const std::unordered_map<const hdk::ir::Node*, size_t>& orig_node_sizes) {
  std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>
      liveouts_renumbering;
  std::vector<const hdk::ir::Node*> ready_nodes;
  AvailabilityChecker checker(liveouts_renumbering, intact_nodes);
  for (auto node : nodes) {
    // Ignore empty live_out due to some invalid node
    if (!does_redef_cols(node.get()) || intact_nodes.count(node.get())) {
      continue;
    }
    auto live_pair = live_outs.find(node.get());
    CHECK(live_pair != live_outs.end());
    auto old_live_outs = live_pair->second;
    add_new_indices_for(
        node.get(), liveouts_renumbering, old_live_outs, intact_nodes, orig_node_sizes);
    if (auto aggregate = std::dynamic_pointer_cast<hdk::ir::Aggregate>(node)) {
      hdk::ir::ExprPtrVector new_exprs;
      auto key_name_it = aggregate->getFields().begin();
      std::vector<std::string> new_fields(key_name_it,
                                          key_name_it + aggregate->getGroupByCount());
      for (size_t i = aggregate->getGroupByCount(), j = 0;
           i < aggregate->getFields().size() && j < aggregate->getAggsCount();
           ++i, ++j) {
        if (old_live_outs.count(i)) {
          new_exprs.push_back(aggregate->getAgg(j));
          new_fields.push_back(aggregate->getFieldName(i));
        }
      }
      aggregate->setAggExprs(std::move(new_exprs));
      aggregate->setFields(std::move(new_fields));
    } else if (auto project = std::dynamic_pointer_cast<hdk::ir::Project>(node)) {
      hdk::ir::ExprPtrVector new_exprs;
      std::vector<std::string> new_fields;
      for (size_t i = 0; i < project->size(); ++i) {
        if (old_live_outs.count(i)) {
          new_exprs.push_back(project->getExpr(i));
          new_fields.push_back(project->getFieldName(i));
        }
      }
      project->setExpressions(std::move(new_exprs));
      project->setFields(std::move(new_fields));
    } else {
      CHECK(false);
    }
    auto usrs_it = du_web.find(node.get());
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (checker.hasAllSrcReady(usr)) {
        ready_nodes.push_back(usr);
      }
    }
  }
  return {liveouts_renumbering, ready_nodes};
}

void propagate_input_renumbering(
    std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>&
        liveout_renumbering,
    const std::vector<const hdk::ir::Node*>& ready_nodes,
    const std::unordered_map<const hdk::ir::Node*, std::unordered_set<size_t>>&
        old_liveouts,
    const std::unordered_set<const hdk::ir::Node*>& intact_nodes,
    const std::unordered_map<const hdk::ir::Node*,
                             std::unordered_set<const hdk::ir::Node*>>& du_web,
    const std::unordered_map<const hdk::ir::Node*, size_t>& orig_node_sizes) {
  InputRenumberVisitor visitor(liveout_renumbering);
  AvailabilityChecker checker(liveout_renumbering, intact_nodes);
  std::deque<const hdk::ir::Node*> work_set(ready_nodes.begin(), ready_nodes.end());
  while (!work_set.empty()) {
    auto walker = work_set.front();
    work_set.pop_front();
    CHECK(!dynamic_cast<const hdk::ir::Scan*>(walker));
    auto node = const_cast<hdk::ir::Node*>(walker);
    if (auto project = dynamic_cast<hdk::ir::Project*>(node)) {
      hdk::ir::ExprPtrVector new_exprs;
      new_exprs.reserve(project->size());
      for (auto& expr : project->getExprs()) {
        new_exprs.emplace_back(visitor.visit(expr.get()));
      }
      project->setExpressions(std::move(new_exprs));
    } else if (auto aggregate = dynamic_cast<hdk::ir::Aggregate*>(node)) {
      auto src_it = liveout_renumbering.find(node->getInput(0));
      CHECK(src_it != liveout_renumbering.end());
      InputSimpleRenumberVisitor<true> visitor(src_it->second);
      hdk::ir::ExprPtrVector new_exprs;
      new_exprs.reserve(aggregate->getAggsCount());
      for (auto& expr : aggregate->getAggs()) {
        new_exprs.emplace_back(visitor.visit(expr.get()));
      }
      aggregate->setAggExprs(std::move(new_exprs));
    } else if (auto join = dynamic_cast<hdk::ir::Join*>(node)) {
      auto new_condition = visitor.visit(join->getCondition());
      join->setCondition(std::move(new_condition));
    } else if (auto filter = dynamic_cast<hdk::ir::Filter*>(node)) {
      auto new_condition_expr = visitor.visit(filter->getConditionExpr());
      filter->setCondition(new_condition_expr);
    } else if (auto sort = dynamic_cast<hdk::ir::Sort*>(node)) {
      auto src_it = liveout_renumbering.find(node->getInput(0));
      CHECK(src_it != liveout_renumbering.end());
      std::vector<hdk::ir::SortField> new_collations;
      for (size_t i = 0; i < sort->collationCount(); ++i) {
        new_collations.push_back(
            renumber_sort_field(sort->getCollation(i), src_it->second));
      }
      sort->setCollation(std::move(new_collations));
    } else if (!dynamic_cast<hdk::ir::LogicalUnion*>(node)) {
      LOG(FATAL) << "Unhandled node type: " << node->toString();
    }

    // Ignore empty live_out due to some invalid node
    if (does_redef_cols(node) || intact_nodes.count(node)) {
      continue;
    }
    auto live_pair = old_liveouts.find(node);
    CHECK(live_pair != old_liveouts.end());
    auto live_out = live_pair->second;
    add_new_indices_for(
        node, liveout_renumbering, live_out, intact_nodes, orig_node_sizes);
    auto usrs_it = du_web.find(walker);
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (checker.hasAllSrcReady(usr)) {
        work_set.push_back(usr);
      }
    }
  }
}

}  // namespace

void eliminate_dead_columns(std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  if (nodes.empty()) {
    return;
  }
  auto root = nodes.back().get();
  if (!root) {
    return;
  }
  CHECK(!dynamic_cast<const hdk::ir::Scan*>(root) &&
        !dynamic_cast<const hdk::ir::Join*>(root));
  // Mark
  auto old_liveouts = mark_live_columns(nodes);
  std::unordered_set<const hdk::ir::Node*> intact_nodes;
  bool has_dead_cols = false;
  for (auto live_pair : old_liveouts) {
    auto node = live_pair.first;
    const auto& outs = live_pair.second;
    if (outs.empty()) {
      LOG(WARNING) << "RA node with no used column: " << node->toString();
      // Ignore empty live_out due to some invalid node
      intact_nodes.insert(node);
    }
    if (any_dead_col_in(node, outs)) {
      has_dead_cols = true;
    } else {
      intact_nodes.insert(node);
    }
  }
  if (!has_dead_cols) {
    return;
  }
  auto web = build_du_web(nodes);
  try_insert_coalesceable_proj(nodes, old_liveouts, web);

  for (auto node : nodes) {
    if (intact_nodes.count(node.get()) || does_redef_cols(node.get())) {
      continue;
    }
    bool intact = true;
    for (size_t i = 0; i < node->inputCount(); ++i) {
      auto source = node->getInput(i);
      if (!dynamic_cast<const hdk::ir::Scan*>(source) && !intact_nodes.count(source)) {
        intact = false;
        break;
      }
    }
    if (intact) {
      intact_nodes.insert(node.get());
    }
  }

  std::unordered_map<const hdk::ir::Node*, size_t> orig_node_sizes;
  for (auto node : nodes) {
    orig_node_sizes.insert(std::make_pair(node.get(), node->size()));
  }
  // Sweep
  std::unordered_map<const hdk::ir::Node*, std::unordered_map<size_t, size_t>>
      liveout_renumbering;
  std::vector<const hdk::ir::Node*> ready_nodes;
  std::tie(liveout_renumbering, ready_nodes) =
      sweep_dead_columns(old_liveouts, nodes, intact_nodes, web, orig_node_sizes);
  // Propagate
  propagate_input_renumbering(
      liveout_renumbering, ready_nodes, old_liveouts, intact_nodes, web, orig_node_sizes);
}

void eliminate_dead_subqueries(
    std::vector<std::shared_ptr<const hdk::ir::ScalarSubquery>>& subqueries,
    hdk::ir::Node const* root) {
  if (!subqueries.empty()) {
    auto live_subqueries = SubQueryCollector::getLiveSubQueries(root);
    int live_count = 0;
    for (size_t i = 0; i < subqueries.size(); ++i) {
      if (live_subqueries.count(subqueries[i]->node())) {
        subqueries[live_count++] = std::move(subqueries[i]);
      }
    }
    subqueries.resize(live_count);
  }
}

namespace {

class InputSinker : public hdk::ir::ExprRewriter {
 public:
  InputSinker(const std::unordered_map<size_t, size_t>& old_to_new_idx,
              const hdk::ir::Node* new_src)
      : old_to_new_in_idx_(old_to_new_idx), target_(new_src) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    CHECK_EQ(target_->inputCount(), size_t(1));
    CHECK_EQ(target_->getInput(0), col_ref->node());
    auto idx_it = old_to_new_in_idx_.find(col_ref->index());
    CHECK(idx_it != old_to_new_in_idx_.end());
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
        getColumnType(target_, idx_it->second), target_, idx_it->second);
  }

 private:
  const std::unordered_map<size_t, size_t>& old_to_new_in_idx_;
  const hdk::ir::Node* target_;
};

class ConditionReplacer : public hdk::ir::ExprRewriter {
 public:
  ConditionReplacer(
      const std::unordered_map<size_t, hdk::ir::ExprPtr>& idx_to_sub_condition)
      : idx_to_subcond_(idx_to_sub_condition) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    auto subcond_it = idx_to_subcond_.find(col_ref->index());
    if (subcond_it != idx_to_subcond_.end()) {
      return visit(subcond_it->second.get());
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

 private:
  const std::unordered_map<size_t, hdk::ir::ExprPtr>& idx_to_subcond_;
};

}  // namespace

void sink_projected_boolean_expr_to_join(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  auto web = build_du_web(nodes);
  auto liveouts = mark_live_columns(nodes);
  for (auto node : nodes) {
    auto project = std::dynamic_pointer_cast<hdk::ir::Project>(node);
    // TODO(miyu): relax hdk::ir::Scan limitation
    if (!project || project->isSimple() ||
        !dynamic_cast<const hdk::ir::Scan*>(project->getInput(0))) {
      continue;
    }
    auto usrs_it = web.find(project.get());
    CHECK(usrs_it != web.end());
    auto& usrs = usrs_it->second;
    if (usrs.size() != 1) {
      continue;
    }
    auto join = dynamic_cast<hdk::ir::Join*>(const_cast<hdk::ir::Node*>(*usrs.begin()));
    if (!join) {
      continue;
    }
    auto outs_it = liveouts.find(join);
    CHECK(outs_it != liveouts.end());
    std::unordered_map<size_t, size_t> in_to_out_index;
    std::unordered_set<size_t> boolean_expr_indicies;
    bool discarded = false;
    for (size_t i = 0; i < project->size(); ++i) {
      auto expr = project->getExpr(i);
      if (expr->type()->isBoolean() &&
          (expr->is<hdk::ir::UOper>() || expr->is<hdk::ir::BinOper>())) {
        boolean_expr_indicies.insert(i);
      } else {
        // TODO(miyu): relax?
        if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(expr.get())) {
          in_to_out_index.insert(std::make_pair(col_ref->index(), i));
        } else {
          discarded = true;
        }
      }
    }
    if (discarded || boolean_expr_indicies.empty()) {
      continue;
    }
    const size_t index_base =
        join->getInput(0) == project.get() ? 0 : join->getInput(0)->size();
    for (auto i : boolean_expr_indicies) {
      auto join_idx = index_base + i;
      if (outs_it->second.count(join_idx)) {
        discarded = true;
        break;
      }
    }
    if (discarded) {
      continue;
    }
    std::vector<size_t> unloaded_input_indices;
    std::unordered_map<size_t, hdk::ir::ExprPtr> in_idx_to_new_subcond;
    // Given all are dead right after join, safe to sink
    for (auto i : boolean_expr_indicies) {
      auto inputs = InputCollector::collect(project->getExpr(i).get(), project.get());
      for (auto& in : inputs) {
        CHECK_EQ(in.first, project->getInput(0));
        if (!in_to_out_index.count(in.second)) {
          auto curr_out_index = project->size() + unloaded_input_indices.size();
          in_to_out_index.insert(std::make_pair(in.second, curr_out_index));
          unloaded_input_indices.push_back(in.second);
        }
        InputSinker sinker(in_to_out_index, project.get());
        in_idx_to_new_subcond.insert(
            std::make_pair(i, sinker.visit(project->getExpr(i).get())));
      }
    }
    if (in_idx_to_new_subcond.empty()) {
      continue;
    }
    hdk::ir::ExprPtrVector new_proj_exprs;
    for (size_t i = 0; i < project->size(); ++i) {
      if (boolean_expr_indicies.count(i)) {
        new_proj_exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            getColumnType(project->getInput(0), 0), project->getInput(0), 0));
      } else {
        CHECK(project->getExpr(i)->is<hdk::ir::ColumnRef>());
        new_proj_exprs.push_back(project->getExpr(i));
      }
    }
    for (auto i : unloaded_input_indices) {
      new_proj_exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          getColumnType(project->getInput(0), i), project->getInput(0), i));
    }
    project->setExpressions(std::move(new_proj_exprs));

    ConditionReplacer replacer(in_idx_to_new_subcond);
    auto new_condition = replacer.visit(join->getCondition());
    join->setCondition(std::move(new_condition));
  }
}

namespace {

class InputRedirector : public hdk::ir::ExprRewriter {
 public:
  InputRedirector(const hdk::ir::Node* old_src, const hdk::ir::Node* new_src)
      : old_src_(old_src), new_src_(new_src) {
    CHECK_NE(old_src_, new_src_);
  }

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    CHECK_EQ(old_src_, col_ref->node());
    auto idx = col_ref->index();
    auto ti = getColumnType(new_src_, idx);
    if (auto join = dynamic_cast<const hdk::ir::Join*>(new_src_)) {
      auto lhs_size = join->getInput(0)->size();
      if (idx >= lhs_size) {
        return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            ti, join->getInput(1), idx - lhs_size);
      } else {
        return hdk::ir::makeExpr<hdk::ir::ColumnRef>(ti, join->getInput(0), idx);
      }
    }

    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(ti, new_src_, idx);
  }

 private:
  const hdk::ir::Node* old_src_;
  const hdk::ir::Node* new_src_;
};

void replace_all_usages(
    std::shared_ptr<const hdk::ir::Node> old_def_node,
    std::shared_ptr<const hdk::ir::Node> new_def_node,
    std::unordered_map<const hdk::ir::Node*, std::shared_ptr<hdk::ir::Node>>&
        deconst_mapping,
    std::unordered_map<const hdk::ir::Node*, std::unordered_set<const hdk::ir::Node*>>&
        du_web) {
  auto usrs_it = du_web.find(old_def_node.get());
  CHECK(usrs_it != du_web.end());
  for (auto usr : usrs_it->second) {
    auto usr_it = deconst_mapping.find(usr);
    CHECK(usr_it != deconst_mapping.end());
    usr_it->second->replaceInput(old_def_node, new_def_node);
  }
  auto new_usrs_it = du_web.find(new_def_node.get());
  CHECK(new_usrs_it != du_web.end());
  new_usrs_it->second.insert(usrs_it->second.begin(), usrs_it->second.end());
  usrs_it->second.clear();
}

}  // namespace

void fold_filters(std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  std::unordered_map<const hdk::ir::Node*, std::shared_ptr<hdk::ir::Node>>
      deconst_mapping;
  for (auto node : nodes) {
    deconst_mapping.insert(std::make_pair(node.get(), node));
  }

  auto web = build_du_web(nodes);
  for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); ++node_it) {
    auto& node = *node_it;
    if (auto filter = std::dynamic_pointer_cast<hdk::ir::Filter>(node)) {
      CHECK_EQ(filter->inputCount(), size_t(1));
      auto src_filter = dynamic_cast<const hdk::ir::Filter*>(filter->getInput(0));
      if (!src_filter) {
        continue;
      }
      auto siblings_it = web.find(src_filter);
      if (siblings_it == web.end() || siblings_it->second.size() != size_t(1)) {
        continue;
      }
      auto src_it = deconst_mapping.find(src_filter);
      CHECK(src_it != deconst_mapping.end());
      auto folded_filter = std::dynamic_pointer_cast<hdk::ir::Filter>(src_it->second);
      CHECK(folded_filter);
      InputRedirector visitor(folded_filter.get(), folded_filter->getInput(0));
      hdk::ir::ExprPtr lhs = folded_filter->getConditionExprShared();
      hdk::ir::ExprPtr rhs = visitor.visit(filter->getConditionExpr());
      bool nullable = lhs->type()->nullable() || rhs->type()->nullable();
      auto new_condition =
          hdk::ir::makeExpr<hdk::ir::BinOper>(lhs->type()->ctx().boolean(nullable),
                                              false,
                                              hdk::ir::OpType::kAnd,
                                              hdk::ir::Qualifier::kOne,
                                              lhs,
                                              rhs);
      folded_filter->setCondition(std::move(new_condition));
      replace_all_usages(filter, folded_filter, deconst_mapping, web);
      deconst_mapping.erase(filter.get());
      web.erase(filter.get());
      web[filter->getInput(0)].erase(filter.get());
      node.reset();
    }
  }

  if (!nodes.empty()) {
    auto sink = nodes.back();
    for (auto node_it = std::next(nodes.rbegin()); node_it != nodes.rend(); ++node_it) {
      if (sink) {
        break;
      }
      sink = *node_it;
    }
    CHECK(sink);
    cleanup_dead_nodes(nodes);
  }
}

namespace {

std::vector<const hdk::ir::Expr*> find_hoistable_conditions(
    const hdk::ir::Expr* condition,
    const hdk::ir::Node* source,
    const size_t first_col_idx,
    const size_t last_col_idx) {
  if (auto bin_op = dynamic_cast<const hdk::ir::BinOper*>(condition)) {
    switch (bin_op->opType()) {
      case hdk::ir::OpType::kAnd: {
        auto lhs_conditions = find_hoistable_conditions(
            bin_op->leftOperand(), source, first_col_idx, last_col_idx);
        auto rhs_conditions = find_hoistable_conditions(
            bin_op->rightOperand(), source, first_col_idx, last_col_idx);
        if (lhs_conditions.size() == 1 && rhs_conditions.size() == 1 &&
            lhs_conditions.front() == bin_op->leftOperand() &&
            rhs_conditions.front() == bin_op->rightOperand()) {
          return {condition};
        }
        lhs_conditions.insert(
            lhs_conditions.end(), rhs_conditions.begin(), rhs_conditions.end());
        return lhs_conditions;
      }
      case hdk::ir::OpType::kEq: {
        auto lhs_conditions = find_hoistable_conditions(
            bin_op->leftOperand(), source, first_col_idx, last_col_idx);
        auto rhs_conditions = find_hoistable_conditions(
            bin_op->rightOperand(), source, first_col_idx, last_col_idx);
        auto lhs_in =
            lhs_conditions.size() == 1
                ? dynamic_cast<const hdk::ir::ColumnRef*>(lhs_conditions.front())
                : nullptr;
        auto rhs_in =
            rhs_conditions.size() == 1
                ? dynamic_cast<const hdk::ir::ColumnRef*>(rhs_conditions.front())
                : nullptr;
        if (lhs_in && rhs_in) {
          return {condition};
        }
        break;
      }
      default:
        break;
    }
  } else if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(condition)) {
    if (col_ref->node() == source) {
      const auto col_idx = col_ref->index();
      return {col_idx >= first_col_idx && col_idx <= last_col_idx ? condition : nullptr};
    }
  }

  return {};
}

class JoinTargetRebaseVisitor : public hdk::ir::ExprRewriter {
 public:
  JoinTargetRebaseVisitor(const hdk::ir::Join* join, const unsigned old_base)
      : join_(join), old_base_(old_base), inp0_size_(join->getInput(0)->size()) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    auto cur_idx = col_ref->index();
    CHECK_GE(cur_idx, old_base_);
    auto new_idx = cur_idx - old_base_;
    auto ti = getColumnType(join_, new_idx);
    if (new_idx >= inp0_size_) {
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          ti, join_->getInput(1), new_idx - inp0_size_);
    }
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(ti, join_->getInput(0), new_idx);
  }

 private:
  const hdk::ir::Join* join_;
  const unsigned old_base_;
  const size_t inp0_size_;
};

hdk::ir::ExprPtr makeConstantExpr(bool val) {
  Datum d;
  d.boolval = val;
  return hdk::ir::makeExpr<hdk::ir::Constant>(
      hdk::ir::Context::defaultCtx().boolean(false), false, d);
}

class SubConditionRemoveVisitor : public hdk::ir::ExprRewriter {
 public:
  SubConditionRemoveVisitor(const std::vector<const hdk::ir::Expr*> sub_conds)
      : sub_conditions_(sub_conds.begin(), sub_conds.end()) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (sub_conditions_.count(col_ref)) {
      return makeConstantExpr(true);
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

  hdk::ir::ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) override {
    if (sub_conditions_.count(bin_oper)) {
      return makeConstantExpr(true);
    }
    return ExprRewriter::visitBinOper(bin_oper);
  }

 private:
  std::unordered_set<const hdk::ir::Expr*> sub_conditions_;
};

}  // namespace

void hoist_filter_cond_to_cross_join(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  std::unordered_set<const hdk::ir::Node*> visited;
  auto web = build_du_web(nodes);
  for (auto node : nodes) {
    if (visited.count(node.get())) {
      continue;
    }
    visited.insert(node.get());
    auto join = dynamic_cast<hdk::ir::Join*>(node.get());
    if (join && join->getJoinType() == JoinType::INNER) {
      // Only allow cross join for now.
      if (auto literal = dynamic_cast<const hdk::ir::Constant*>(join->getCondition())) {
        // Assume Calcite always generates an inner join on constant boolean true for
        // cross join.
        CHECK(literal->type()->isBoolean() && literal->intVal());
        size_t first_col_idx = 0;
        const hdk::ir::Filter* filter = nullptr;
        std::vector<const hdk::ir::Join*> join_seq{join};
        for (const hdk::ir::Join* curr_join = join; !filter;) {
          auto usrs_it = web.find(curr_join);
          CHECK(usrs_it != web.end());
          if (usrs_it->second.size() != size_t(1)) {
            break;
          }
          auto only_usr = *usrs_it->second.begin();
          if (auto usr_join = dynamic_cast<const hdk::ir::Join*>(only_usr)) {
            if (join == usr_join->getInput(1)) {
              const auto src1_offset = usr_join->getInput(0)->size();
              first_col_idx += src1_offset;
            }
            join_seq.push_back(usr_join);
            curr_join = usr_join;
            continue;
          }

          filter = dynamic_cast<const hdk::ir::Filter*>(only_usr);
          break;
        }
        if (!filter) {
          visited.insert(join_seq.begin(), join_seq.end());
          continue;
        }
        const auto src_join = dynamic_cast<const hdk::ir::Join*>(filter->getInput(0));
        CHECK(src_join);
        auto modified_filter = const_cast<hdk::ir::Filter*>(filter);

        if (src_join == join) {
          auto filter_condition = modified_filter->getConditionExprShared();
          modified_filter->setCondition(makeConstantExpr(true));
          join->setCondition(std::move(filter_condition));
          continue;
        }
        const auto src1_base = src_join->getInput(0)->size();
        auto source =
            first_col_idx < src1_base ? src_join->getInput(0) : src_join->getInput(1);
        first_col_idx =
            first_col_idx < src1_base ? first_col_idx : first_col_idx - src1_base;
        auto join_conditions =
            find_hoistable_conditions(filter->getConditionExpr(),
                                      source,
                                      first_col_idx,
                                      first_col_idx + join->size() - 1);
        if (join_conditions.empty()) {
          continue;
        }

        JoinTargetRebaseVisitor visitor(join, first_col_idx);
        if (join_conditions.size() == 1) {
          auto new_join_condition = visitor.visit(join_conditions.front());
          join->setCondition(new_join_condition);
        } else {
          auto new_join_condition = visitor.visit(join_conditions[0]);
          for (size_t i = 1; i < join_conditions.size(); ++i) {
            auto rhs = visitor.visit(join_conditions[i]);
            auto res_type = rhs->type()->ctx().boolean(
                rhs->type()->nullable() || new_join_condition->type()->nullable());
            new_join_condition =
                hdk::ir::makeExpr<hdk::ir::BinOper>(res_type,
                                                    hdk::ir::OpType::kAnd,
                                                    hdk::ir::Qualifier::kOne,
                                                    new_join_condition,
                                                    rhs);
          }
          join->setCondition(new_join_condition);
        }

        SubConditionRemoveVisitor remover(join_conditions);
        auto new_filter_expr = remover.visit(filter->getConditionExpr());
        modified_filter->setCondition(new_filter_expr);
      }
    }
  }
}

void sync_field_names_if_necessary(std::shared_ptr<const hdk::ir::Project> from_node,
                                   hdk::ir::Node* to_node) noexcept {
  auto from_fields = from_node->getFields();
  if (!from_fields.empty()) {
    if (auto proj_to = dynamic_cast<hdk::ir::Project*>(to_node);
        proj_to && proj_to->getFields().size() == from_fields.size()) {
      proj_to->setFields(std::move(from_fields));
    } else if (auto agg_to = dynamic_cast<hdk::ir::Aggregate*>(to_node);
               agg_to && agg_to->getFields().size() == from_fields.size()) {
      agg_to->setFields(std::move(from_fields));
    }
  }
}

// For some reason, Calcite generates Sort, Project, Sort sequences where the
// two Sort nodes are identical and the Project is identity. Simplify this
// pattern by re-binding the input of the second sort to the input of the first.
void simplify_sort(std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  if (nodes.size() < 3) {
    return;
  }
  for (size_t i = 0; i <= nodes.size() - 3;) {
    auto first_sort = std::dynamic_pointer_cast<hdk::ir::Sort>(nodes[i]);
    const auto project = std::dynamic_pointer_cast<const hdk::ir::Project>(nodes[i + 1]);
    auto second_sort = std::dynamic_pointer_cast<hdk::ir::Sort>(nodes[i + 2]);
    if (first_sort && second_sort && project && project->isIdentity() &&
        *first_sort == *second_sort) {
      sync_field_names_if_necessary(project, /* an input of the second sort */
                                    const_cast<hdk::ir::Node*>(first_sort->getInput(0)));
      second_sort->replaceInput(second_sort->getAndOwnInput(0),
                                first_sort->getAndOwnInput(0));
      nodes[i].reset();
      nodes[i + 1].reset();
      i += 3;
    } else {
      ++i;
    }
  }

  std::vector<std::shared_ptr<hdk::ir::Node>> new_nodes;
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    new_nodes.push_back(node);
  }
  nodes.swap(new_nodes);
}
