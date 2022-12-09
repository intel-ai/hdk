/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryBuilder.h"
#include "ExprCollector.h"
#include "ExprRewriter.h"

#include "Shared/SqlTypesLayout.h"

#include <boost/algorithm/string.hpp>

namespace hdk::ir {

namespace {

int normalizeColIndex(const Node* node, int col_idx) {
  int size = static_cast<int>(node->size());
  if (col_idx >= size || col_idx < -size) {
    throw InvalidQueryError() << "Out-of-border column index.\n"
                              << "  Column index: " << col_idx << "\n"
                              << "  Node: " << node->toString() << "\n";
  }
  return col_idx < 0 ? size + col_idx : col_idx;
}

std::string getFieldName(const Node* node, int col_idx) {
  col_idx = normalizeColIndex(node, col_idx);
  if (auto scan = node->as<Scan>()) {
    return scan->getColumnInfo(col_idx)->name;
  }
  if (auto proj = node->as<Project>()) {
    return proj->getFieldName(col_idx);
  }
  if (auto agg = node->as<Aggregate>()) {
    return agg->getFieldName(col_idx);
  }

  throw InvalidQueryError() << "getFieldName error: unsupported node: "
                            << node->toString();
}

ExprPtr getRefByName(const Scan* scan, const std::string& col_name) {
  for (size_t i = 0; i < scan->size(); ++i) {
    if (scan->getColumnInfo(i)->name == col_name) {
      return getNodeColumnRef(scan, (unsigned)i);
    }
  }
  return nullptr;
}

ExprPtr getRefByName(const Project* proj, const std::string& col_name) {
  for (size_t i = 0; i < proj->size(); ++i) {
    if (proj->getFieldName(i) == col_name) {
      return getNodeColumnRef(proj, (unsigned)i);
    }
  }
  return nullptr;
}

ExprPtr getRefByName(const Aggregate* agg, const std::string& col_name) {
  for (size_t i = 0; i < agg->size(); ++i) {
    if (agg->getFieldName(i) == col_name) {
      return getNodeColumnRef(agg, (unsigned)i);
    }
  }
  return nullptr;
}

ExprPtr getRefByName(const Node* node, const std::string& col_name) {
  ExprPtr res = nullptr;
  if (auto scan = node->as<Scan>()) {
    res = getRefByName(scan, col_name);
  } else if (auto proj = node->as<Project>()) {
    res = getRefByName(proj, col_name);
  } else if (auto agg = node->as<Aggregate>()) {
    res = getRefByName(agg, col_name);
  } else {
    throw InvalidQueryError() << "getRefByName error: unsupported node: "
                              << node->toString();
  }

  if (!res) {
    throw InvalidQueryError() << "getRefByName error: unknown column name.\n"
                              << "  Column name: " << col_name << "\n"
                              << "  Node: " << node->toString() << "\n";
  }

  return res;
}

ExprPtr getRefByIndex(const Node* node, int col_idx) {
  col_idx = normalizeColIndex(node, col_idx);
  return getNodeColumnRef(node, col_idx);
}

std::string chooseName(const std::string& name,
                       const std::unordered_set<std::string>& names) {
  std::string prefix;
  if (name.empty()) {
    prefix = "expr_";
  } else if (names.count(name)) {
    prefix = name + "_";
  } else {
    return name;
  }

  size_t idx = 1;
  std::string candidate;
  do {
    candidate = prefix + std::to_string(idx++);
  } while (names.count(candidate));

  return candidate;
}

std::vector<std::string> buildFieldNames(const std::vector<BuilderExpr>& exprs) {
  std::unordered_set<std::string> names;
  // First check all manually set names are unique.
  for (auto& expr : exprs) {
    if (!expr.isAutoNamed()) {
      if (expr.name().empty()) {
        throw InvalidQueryError() << "Empty field names are not allowed";
      }
      auto pr = names.insert(expr.name());
      if (!pr.second) {
        throw InvalidQueryError() << "Duplicated field name: " << expr.name();
      }
    }
  }

  // Build the resulting vector adding suffixes to auto-names when needed.
  std::vector<std::string> res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    if (expr.isAutoNamed()) {
      auto name = chooseName(expr.name(), names);
      auto pr = names.insert(name);
      CHECK(pr.second);
      res.emplace_back(std::move(name));
    } else {
      res.emplace_back(expr.name());
    }
  }

  return res;
}

ExprPtrVector collectExprs(const std::vector<BuilderExpr>& exprs) {
  ExprPtrVector res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    res.emplace_back(expr.expr());
  }
  return res;
}

class InputNodesCollector
    : public ExprCollector<std::unordered_set<const Node*>, InputNodesCollector> {
 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    result_.insert(col_ref->node());
  }
};

void checkExprInput(const ExprPtr& expr,
                    const std::unordered_set<const Node*>& allowed_nodes,
                    const std::string& node_name) {
  auto expr_nodes = InputNodesCollector::collect(expr);
  for (auto& node : expr_nodes) {
    if (!allowed_nodes.count(node)) {
      std::stringstream ss;
      ss << "Wrong expression in a " << node_name
         << ": non-input node is referenced by an expression." << std::endl
         << "  Expression: " << expr->toString() << std::endl
         << "  Referenced node: " << node->toString() << std::endl
         << "  Input nodes:" << std::endl;
      for (auto node : allowed_nodes) {
        ss << "    " << node->toString() << std::endl;
      }
      throw InvalidQueryError(ss.str());
    }
  }
}

void checkExprInput(const ExprPtrVector& exprs,
                    const std::vector<const Node*>& nodes,
                    const std::string& node_name) {
  std::unordered_set<const Node*> allowed(nodes.begin(), nodes.end());
  for (auto& expr : exprs) {
    checkExprInput(expr, allowed, node_name);
  }
}

void checkExprInput(const std::vector<BuilderExpr>& exprs,
                    const std::vector<const Node*>& nodes,
                    const std::string& node_name) {
  std::unordered_set<const Node*> allowed(nodes.begin(), nodes.end());
  for (auto& expr : exprs) {
    checkExprInput(expr.expr(), allowed, node_name);
  }
}

bool isIdentShuffle(const std::vector<int>& shuffle) {
  for (int i = 0; i < (int)shuffle.size(); ++i) {
    if (i != shuffle[i]) {
      return false;
    }
  }
  return true;
}

class InputRewriter : public ExprRewriter {
 public:
  InputRewriter(const hdk::ir::Node* new_base, const std::vector<int>& new_indexes)
      : new_base_(new_base), new_indexes_(new_indexes) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    CHECK_LT((size_t)col_ref->index(), new_indexes_.size());
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
        col_ref->type(), new_base_, new_indexes_[col_ref->index()]);
  }

 private:
  const hdk::ir::Node* new_base_;
  const std::vector<int>& new_indexes_;
};

std::vector<BuilderExpr> replaceInput(const std::vector<BuilderExpr>& exprs,
                                      const Node* new_base,
                                      const std::vector<int> new_indexes) {
  InputRewriter rewriter(new_base, new_indexes);
  std::vector<BuilderExpr> res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    res.emplace_back(expr.rewrite(rewriter));
  }
  return res;
}

void checkCstArrayType(const Type* type, size_t elems) {
  if (!type->isArray()) {
    throw InvalidQueryError()
        << "Only array types can be used to translate a vector to a literal. Provided: "
        << type->toString();
  }
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  // Only few types are actually supported by codegen.
  if (!elem_type->isInt8() && !elem_type->isInt32() && !elem_type->isFp64()) {
    throw InvalidQueryError() << "Only int8, int32, and fp64 elements are supported in "
                                 "array literals. Requested: "
                              << elem_type->toString();
  }
  if (type->isFixedLenArray()) {
    auto num_elems = type->as<FixedLenArrayType>()->numElems();
    if (static_cast<size_t>(num_elems) != elems) {
      throw InvalidQueryError()
          << "Literal array elements count mismatch. Expected " << num_elems
          << " elements, provided " << elems << " elements.";
    }
  }
}

}  // namespace

BuilderExpr::BuilderExpr(const QueryBuilder& builder,
                         ExprPtr expr,
                         const std::string& name,
                         bool auto_name)
    : builder_(builder), expr_(expr), name_(name), auto_name_(auto_name) {}

BuilderExpr BuilderExpr::name(const std::string& name) const {
  return {builder_, expr_, name, false};
}

BuilderExpr BuilderExpr::avg() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for avg aggregate: "
                              << expr_->type()->toString();
  }
  auto agg =
      makeExpr<AggExpr>(builder_.ctx_.fp64(), AggType::kAvg, expr_, false, nullptr);
  auto name = name_.empty() ? "avg" : name_ + "_avg";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::min() const {
  if (!expr_->type()->isNumber() && !expr_->type()->isDateTime()) {
    throw InvalidQueryError() << "Unsupported type for min aggregate: "
                              << expr_->type()->toString();
  }
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kMin, expr_, false, nullptr);
  auto name = name_.empty() ? "min" : name_ + "_min";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::max() const {
  if (!expr_->type()->isNumber() && !expr_->type()->isDateTime()) {
    throw InvalidQueryError() << "Unsupported type for max aggregate: "
                              << expr_->type()->toString();
  }
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kMax, expr_, false, nullptr);
  auto name = name_.empty() ? "max" : name_ + "_max";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::sum() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for sum aggregate: "
                              << expr_->type()->toString();
  }
  auto res_type = expr_->type();
  if (res_type->isInteger() && res_type->size() < 8) {
    res_type = builder_.ctx_.int64(res_type->nullable());
  }
  auto agg = makeExpr<AggExpr>(res_type, AggType::kSum, expr_, false, nullptr);
  auto name = name_.empty() ? "sum" : name_ + "_sum";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::count(bool is_distinct) const {
  if (!expr_->is<hdk::ir::ColumnRef>()) {
    throw InvalidQueryError()
        << "Count method is valid for column references only. Used for: "
        << expr_->toString();
  }
  auto count_type = builder_.config_->exec.group_by.bigint_count
                        ? builder_.ctx_.int64(false)
                        : builder_.ctx_.int32(false);
  auto agg = makeExpr<AggExpr>(count_type, AggType::kCount, expr_, is_distinct, nullptr);
  auto name = name_.empty() ? "count" : name_ + "_count";
  if (is_distinct) {
    name += "_dist";
  }
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::approxCountDist() const {
  if (!expr_->is<hdk::ir::ColumnRef>()) {
    throw InvalidQueryError()
        << "ApproxCountDist method is valid for column references only. Used for: "
        << expr_->toString();
  }
  auto count_type = builder_.config_->exec.group_by.bigint_count
                        ? builder_.ctx_.int64(false)
                        : builder_.ctx_.int32(false);
  auto agg =
      makeExpr<AggExpr>(count_type, AggType::kApproxCountDistinct, expr_, true, nullptr);
  auto name = name_.empty() ? "approx_count_dist" : name_ + "_approx_count_dist";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::approxQuantile(double val) const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for sum aggregate: "
                              << expr_->type()->toString();
  }
  if (val < 0.0 || val > 1.0) {
    throw InvalidQueryError()
        << "ApproxQuantile expects argument between 0.0 and 1.0 but got " << val;
  }
  Datum d;
  d.doubleval = val;
  auto cst = makeExpr<Constant>(builder_.ctx_.fp64(), false, d);
  auto agg = makeExpr<AggExpr>(
      builder_.ctx_.fp64(), AggType::kApproxQuantile, expr_, false, cst);
  auto name = name_.empty() ? "approx_quantile" : name_ + "_approx_quantile";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::sample() const {
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kSample, expr_, false, nullptr);
  auto name = name_.empty() ? "sample" : name_ + "_sample";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::singleValue() const {
  if (expr_->type()->isVarLen()) {
    throw InvalidQueryError() << "Varlen type " << expr_->type()->toString()
                              << " is not suported for single value aggregate.";
  }
  auto agg =
      makeExpr<AggExpr>(expr_->type(), AggType::kSingleValue, expr_, false, nullptr);
  auto name = name_.empty() ? "single_value" : name_ + "_single_value";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::agg(const std::string& agg_str, double val) const {
  static const std::unordered_map<std::string, AggType> agg_names = {
      {"count", AggType::kCount},
      {"count_dist", AggType::kCount},
      {"count_distinct", AggType::kCount},
      {"count dist", AggType::kCount},
      {"count distinct", AggType::kCount},
      {"sum", AggType::kSum},
      {"min", AggType::kMin},
      {"max", AggType::kMax},
      {"avg", AggType::kAvg},
      {"approx_count_dist", AggType::kApproxCountDistinct},
      {"approx_count_distinct", AggType::kApproxCountDistinct},
      {"approx count dist", AggType::kApproxCountDistinct},
      {"approx count distinct", AggType::kApproxCountDistinct},
      {"approx_quantile", AggType::kApproxQuantile},
      {"approx quantile", AggType::kApproxQuantile},
      {"sample", AggType::kSample},
      {"single_value", AggType::kSingleValue},
      {"single value", AggType::kSingleValue}};
  static const std::unordered_set<std::string> distinct_names = {
      "count_dist", "count_distinct", "count dist", "count distinct"};
  auto agg_str_lower = boost::algorithm::to_lower_copy(agg_str);
  if (!agg_names.count(agg_str_lower)) {
    throw InvalidQueryError() << "Unknown aggregate name: " << agg_str;
  }

  auto kind = agg_names.at(agg_str_lower);
  if (kind == AggType::kApproxQuantile && val == HUGE_VAL) {
    throw InvalidQueryError("Missing argument for approximate quantile aggregate.");
  }

  auto is_distinct = distinct_names.count(agg_str_lower);
  return agg(kind, is_distinct, val);
}

BuilderExpr BuilderExpr::agg(AggType agg_kind, double val) const {
  return agg(agg_kind, false, val);
}

BuilderExpr BuilderExpr::agg(AggType agg_kind, bool is_distinct, double val) const {
  if (is_distinct && agg_kind != AggType::kCount) {
    throw InvalidQueryError() << "Distinct property cannot be set to true for "
                              << agg_kind << " aggregate.";
  }
  if (val != HUGE_VAL && agg_kind != AggType::kApproxQuantile) {
    throw InvalidQueryError() << "Aggregate argument is supported for approximate "
                                 "quantile only but provided for "
                              << agg_kind;
  }

  switch (agg_kind) {
    case AggType::kAvg:
      return avg();
    case AggType::kMin:
      return min();
    case AggType::kMax:
      return max();
    case AggType::kSum:
      return sum();
    case AggType::kCount:
      return count(is_distinct);
    case AggType::kApproxCountDistinct:
      return approxCountDist();
    case AggType::kApproxQuantile:
      return approxQuantile(val);
    case AggType::kSample:
      return sample();
    case AggType::kSingleValue:
      return singleValue();
    default:
      break;
  }
  throw InvalidQueryError() << "Unsupported aggregate type: " << agg_kind;
}

BuilderExpr BuilderExpr::extract(DateExtractField field) const {
  if (!expr_->type()->isDateTime()) {
    throw InvalidQueryError()
        << "Only datetime types are allowed for extract operation. Actual type: "
        << expr_->type()->toString();
  }
  auto extract_expr =
      makeExpr<ExtractExpr>(builder_.ctx_.int64(expr_->type()->nullable()),
                            expr_->containsAgg(),
                            field,
                            expr_->decompress());
  return {builder_, extract_expr, "", true};
}

BuilderExpr BuilderExpr::extract(const std::string& field) const {
  static const std::unordered_map<std::string, DateExtractField> allowed_values = {
      {"year", DateExtractField::kYear},
      {"quarter", DateExtractField::kQuarter},
      {"month", DateExtractField::kMonth},
      {"day", DateExtractField::kDay},
      {"hour", DateExtractField::kHour},
      {"min", DateExtractField::kMinute},
      {"minute", DateExtractField::kMinute},
      {"sec", DateExtractField::kSecond},
      {"second", DateExtractField::kSecond},
      {"milli", DateExtractField::kMilli},
      {"millisecond", DateExtractField::kMilli},
      {"micro", DateExtractField::kMicro},
      {"microsecond", DateExtractField::kMicro},
      {"nano", DateExtractField::kNano},
      {"nanosecond", DateExtractField::kNano},
      {"dow", DateExtractField::kDayOfWeek},
      {"dayofweek", DateExtractField::kDayOfWeek},
      {"day_of_week", DateExtractField::kDayOfWeek},
      {"day of week", DateExtractField::kDayOfWeek},
      {"isodow", DateExtractField::kIsoDayOfWeek},
      {"isodayofweek", DateExtractField::kIsoDayOfWeek},
      {"iso_day_of_week", DateExtractField::kIsoDayOfWeek},
      {"iso day of week", DateExtractField::kIsoDayOfWeek},
      {"doy", DateExtractField::kDayOfYear},
      {"dayofyear", DateExtractField::kDayOfYear},
      {"day_of_year", DateExtractField::kDayOfYear},
      {"day of year", DateExtractField::kDayOfYear},
      {"epoch", DateExtractField::kEpoch},
      {"quarterday", DateExtractField::kQuarterDay},
      {"quarter_day", DateExtractField::kQuarterDay},
      {"quarter day", DateExtractField::kQuarterDay},
      {"week", DateExtractField::kWeek},
      {"weeksunday", DateExtractField::kWeekSunday},
      {"week_sunday", DateExtractField::kWeekSunday},
      {"week sunday", DateExtractField::kWeekSunday},
      {"weeksaturday", DateExtractField::kWeekSaturday},
      {"week_saturday", DateExtractField::kWeekSaturday},
      {"week saturday", DateExtractField::kWeekSaturday},
      {"dateepoch", DateExtractField::kDateEpoch},
      {"date_epoch", DateExtractField::kDateEpoch},
      {"date epoch", DateExtractField::kDateEpoch}};
  auto canonical = boost::trim_copy(boost::to_lower_copy(field));
  if (!allowed_values.count(canonical)) {
    throw InvalidQueryError() << "Cannot parse date extract field: '" << field << "'";
  }
  return extract(allowed_values.at(canonical));
}

BuilderExpr BuilderExpr::cast(const Type* new_type) const {
  if (expr_->type()->isInteger()) {
    if (new_type->isNumber() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isBoolean()) {
      return ne(builder_.cst(0, expr_->type()));
    }
  } else if (expr_->type()->isFloatingPoint()) {
    if (new_type->isInteger() || new_type->isFloatingPoint()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isDecimal()) {
    if (new_type->isNumber() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isBoolean()) {
      return ne(builder_.cst(0, expr_->type()));
    }
  } else if (expr_->type()->isBoolean()) {
    if (new_type->isInteger() || new_type->isDecimal()) {
      std::list<std::pair<ExprPtr, ExprPtr>> expr_list;
      expr_list.emplace_back(expr_, builder_.cst(1, new_type).expr());
      auto case_expr = std::make_shared<CaseExpr>(
          new_type, expr_->containsAgg(), expr_list, builder_.cst(0, new_type).expr());
      if (expr_->type()->nullable()) {
        auto is_null =
            std::make_shared<UOper>(builder_.ctx_.boolean(false), OpType::kIsNull, expr_);
        auto is_not_null =
            std::make_shared<UOper>(builder_.ctx_.boolean(false), OpType::kNot, is_null);
        auto null_cst = std::make_shared<Constant>(new_type, true, Datum{});
        std::list<std::pair<ExprPtr, ExprPtr>> expr_list;
        expr_list.emplace_back(is_not_null, case_expr);
        case_expr =
            makeExpr<CaseExpr>(new_type, expr_->containsAgg(), expr_list, null_cst);
      }
      return {builder_, case_expr, "", true};
    } else if (new_type->isFloatingPoint() || new_type->isBoolean()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isString()) {
    if (new_type->equal(expr_->type()->withNullable(new_type->nullable()))) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isExtDictionary()) {
      if (new_type->as<ExtDictionaryType>()->dictId() <= TRANSIENT_DICT_ID &&
          !expr_->is<Constant>()) {
        throw InvalidQueryError(
            "Cannot apply transient dictionary encoding to non-literal expression.");
      }
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isNumber() || new_type->isDateTime() || new_type->isBoolean() ||
               new_type->isString()) {
      if (expr_->is<Constant>()) {
        try {
          return {builder_, expr_->cast(new_type), "", true};
        } catch (std::runtime_error& e) {
          throw InvalidQueryError(e.what());
        }
      } else {
        throw InvalidQueryError(
            "String conversions for non-literals are not yet supported.");
      }
    }
  } else if (expr_->type()->isExtDictionary()) {
    if (new_type->isText()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isExtDictionary()) {
      if (new_type->as<ExtDictionaryType>()->dictId() <= TRANSIENT_DICT_ID &&
          !expr_->is<Constant>()) {
        throw InvalidQueryError(
            "Cannot apply transient dictionary encoding to non-literal expression.");
      }
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isDate()) {
    if (new_type->isDate() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isTime()) {
    if (new_type->isTime()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isTimestamp()) {
    if (new_type->isNumber() || new_type->isDate() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  }

  throw InvalidQueryError() << "Conversion from " << expr_->type()->toString() << " to "
                            << new_type->toString() << " is not supported.";
}

BuilderExpr BuilderExpr::cast(const std::string& new_type) const {
  return cast(builder_.ctx_.typeFromString(new_type));
}

BuilderExpr BuilderExpr::ne(const BuilderExpr& rhs) const {
  // TODO: add auto-casts?
  if (!expr_->type()->equal(rhs.expr()->type()) &&
      !expr_->type()
           ->withNullable(rhs.expr()->type()->nullable())
           ->equal(rhs.expr()->type())) {
    throw InvalidQueryError() << "Mismatched type for comparison:\n  LHS type: "
                              << expr_->type()->toString()
                              << "\n  RHS type: " << rhs.expr()->type()->toString();
  }
  auto nullable = expr_->type()->nullable() || rhs.expr()->type()->nullable();
  auto bin_oper = makeExpr<BinOper>(
      builder_.ctx_.boolean(nullable), OpType::kNe, Qualifier::kOne, expr_, rhs.expr());
  return {builder_, bin_oper, "", true};
}

BuilderExpr BuilderExpr::logicalNot() const {
  if (!expr_->type()->isBoolean()) {
    throw InvalidQueryError("Only boolean expressions are allowed for NOT operation.");
  }
  if (expr_->is<Constant>()) {
    return builder_.cst(!expr_->as<Constant>()->intVal(), expr_->type());
  }
  auto uoper = makeExpr<UOper>(builder_.ctx_.boolean(expr_->type()->nullable()),
                               expr_->containsAgg(),
                               OpType::kNot,
                               expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::uminus() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError("Only numeric expressions are allowed for UMINUS operation.");
  }
  if (expr_->is<Constant>()) {
    auto cst_expr = expr_->as<Constant>();
    if (cst_expr->type()->isInteger()) {
      return builder_.cst(-cst_expr->intVal(), cst_expr->type());
    } else if (cst_expr->type()->isFloatingPoint()) {
      return builder_.cst(-cst_expr->fpVal(), cst_expr->type());
    } else {
      CHECK(cst_expr->type()->isDecimal());
      return builder_.cstNoScale(-cst_expr->intVal(), cst_expr->type());
    }
  }
  auto uoper =
      makeExpr<UOper>(expr_->type(), expr_->containsAgg(), OpType::kUMinus, expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::rewrite(ExprRewriter& rewriter) const {
  return {builder_, rewriter.visit(expr_.get()), name_, auto_name_};
}

BuilderExpr BuilderExpr::operator!() const {
  return logicalNot();
}

BuilderExpr BuilderExpr::operator-() const {
  return uminus();
}

BuilderSortField::BuilderSortField(int col_idx,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(col_idx), dir_(dir), null_pos_(null_pos) {}

BuilderSortField::BuilderSortField(int col_idx,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(col_idx)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {}

BuilderSortField::BuilderSortField(const std::string& col_name,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(col_name), dir_(dir), null_pos_(null_pos) {}

BuilderSortField::BuilderSortField(const std::string& col_name,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(col_name)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {}

BuilderSortField::BuilderSortField(BuilderExpr expr,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(expr), dir_(dir), null_pos_(null_pos) {
  if (!expr.expr()->is<ColumnRef>()) {
    throw InvalidQueryError() << "Only column references are allowed for "
                                 "sort operation. Provided expression: "
                              << expr.expr()->toString();
  }
}

BuilderSortField::BuilderSortField(BuilderExpr expr,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(expr)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {
  if (!expr.expr()->is<ColumnRef>()) {
    throw InvalidQueryError() << "Only column references are allowed for "
                                 "sort operation. Provided expression: "
                              << expr.expr()->toString();
  }
}

SortDirection BuilderSortField::parseSortDirection(const std::string& val) {
  auto val_lowered = boost::trim_copy(boost::to_lower_copy(val));
  if (val_lowered == "asc" || val_lowered == "ascending") {
    return SortDirection::Ascending;
  } else if (val_lowered == "desc" || val_lowered == "descending") {
    return SortDirection::Descending;
  }
  throw InvalidQueryError() << "Cannot parse sort direction (use 'asc' or 'desc'): '"
                            << val << "'";
}

NullSortedPosition BuilderSortField::parseNullPosition(const std::string& val) {
  auto val_lowered = boost::trim_copy(boost::to_lower_copy(val));
  if (val_lowered == "first") {
    return NullSortedPosition::First;
  } else if (val_lowered == "last") {
    return NullSortedPosition::Last;
  }
  throw InvalidQueryError() << "Cannot parse nulls position (use 'first' or 'last'): '"
                            << val << "'";
}

BuilderNode::BuilderNode(const QueryBuilder& builder, NodePtr node)
    : builder_(builder), node_(node) {}

BuilderExpr BuilderNode::ref(int col_idx) const {
  auto expr = getRefByIndex(node_.get(), col_idx);
  auto name = getFieldName(node_.get(), col_idx);
  return {builder_, expr, name};
}

BuilderExpr BuilderNode::ref(const std::string& col_name) const {
  auto expr = getRefByName(node_.get(), col_name);
  auto name = getFieldName(node_.get(), expr->as<ColumnRef>()->index());
  return {builder_, expr, name};
}

std::vector<BuilderExpr> BuilderNode::ref(std::initializer_list<int> col_indices) const {
  return ref(std::vector<int>(col_indices));
}

std::vector<BuilderExpr> BuilderNode::ref(std::vector<int> col_indices) const {
  std::vector<BuilderExpr> res;
  res.reserve(col_indices.size());
  for (auto col_idx : col_indices) {
    res.emplace_back(ref(col_idx));
  }
  return res;
}

std::vector<BuilderExpr> BuilderNode::ref(
    std::initializer_list<std::string> col_names) const {
  return ref(std::vector<std::string>(col_names));
}

std::vector<BuilderExpr> BuilderNode::ref(std::vector<std::string> col_names) const {
  std::vector<BuilderExpr> res;
  res.reserve(col_names.size());
  for (auto col_name : col_names) {
    res.emplace_back(ref(col_name));
  }
  return res;
}

BuilderExpr BuilderNode::count() const {
  return builder_.count();
}

BuilderExpr BuilderNode::count(int col_idx, bool is_distinct) const {
  return ref(col_idx).count(is_distinct);
}

BuilderExpr BuilderNode::count(const std::string& col_name, bool is_distinct) const {
  return ref(col_name).count(is_distinct);
}

BuilderExpr BuilderNode::count(BuilderExpr col_ref, bool is_distinct) const {
  return col_ref.count(is_distinct);
}

BuilderNode BuilderNode::proj(std::initializer_list<int> col_indices) const {
  return proj(ref(col_indices));
}

BuilderNode BuilderNode::proj(std::initializer_list<int> col_indices,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_indices), fields);
}

BuilderNode BuilderNode::proj(std::initializer_list<std::string> col_names) const {
  return proj(ref(col_names));
}

BuilderNode BuilderNode::proj(std::initializer_list<std::string> col_names,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_names), fields);
}

BuilderNode BuilderNode::proj(int col_idx) const {
  return proj(ref({col_idx}));
}

BuilderNode BuilderNode::proj(int col_idx, const std::string& field_name) const {
  return proj(ref(col_idx), field_name);
}

BuilderNode BuilderNode::proj(const std::vector<int> col_indices) const {
  return proj(ref(col_indices));
}

BuilderNode BuilderNode::proj(const std::vector<int> col_indices,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_indices), fields);
}

BuilderNode BuilderNode::proj(const std::string& col_name) const {
  return proj(ref(col_name));
}

BuilderNode BuilderNode::proj(const std::string& col_name,
                              const std::string& field_name) const {
  return proj(ref(col_name), field_name);
}

BuilderNode BuilderNode::proj(const std::vector<std::string>& col_names) const {
  return proj(ref(col_names));
}

BuilderNode BuilderNode::proj(const std::vector<std::string>& col_names,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_names), fields);
}

BuilderNode BuilderNode::proj(const BuilderExpr& expr) const {
  return proj(std::vector<BuilderExpr>({expr}));
}

BuilderNode BuilderNode::proj(const BuilderExpr& expr, const std::string& field) const {
  return proj(std::vector<BuilderExpr>({expr}), {field});
}

BuilderNode BuilderNode::proj(const std::vector<BuilderExpr>& exprs) const {
  auto fields = buildFieldNames(exprs);
  auto expr_ptrs = collectExprs(exprs);
  return proj(expr_ptrs, fields);
}

BuilderNode BuilderNode::proj(const std::vector<BuilderExpr>& exprs,
                              const std::vector<std::string>& fields) const {
  std::unordered_set<std::string> names;
  for (auto& name : fields) {
    auto pr = names.insert(name);
    if (!pr.second) {
      throw InvalidQueryError() << "Duplicated field name: " << name;
    }
  }
  auto expr_ptrs = collectExprs(exprs);
  return proj(expr_ptrs, fields);
}

BuilderNode BuilderNode::proj() const {
  std::vector<int> col_indexes(node_->size());
  for (int i = 0; i < (int)col_indexes.size(); ++i) {
    col_indexes[i] = i;
  }
  return proj(col_indexes);
}

BuilderNode BuilderNode::proj(const ExprPtrVector& exprs,
                              const std::vector<std::string>& fields) const {
  if (exprs.empty()) {
    throw InvalidQueryError(
        "Empty projections are not allowed. At least one expression is required.");
  }
  if (exprs.size() != fields.size()) {
    throw InvalidQueryError() << "Mismathed number of expressions (" << exprs.size()
                              << ") and field names (" << fields.size() << ")";
  }
  checkExprInput(exprs, {node_.get()}, "projection");
  auto proj = std::make_shared<Project>(exprs, fields, node_);
  return {builder_, proj};
}

BuilderExpr BuilderNode::parseAggString(const std::string& agg_str) const {
  auto agg_str_lower = boost::trim_copy(boost::algorithm::to_lower_copy(agg_str));
  if (agg_str_lower == "count") {
    return count();
  }

  // Parse string like <agg_name>(<col_name>[, <agg_param>]).
  auto pos = agg_str_lower.find('(');
  if (agg_str_lower.back() == ')' && pos != std::string::npos) {
    auto agg_name = boost::trim_copy(agg_str_lower.substr(0, pos));
    auto col_name =
        boost::trim_copy(agg_str_lower.substr(pos + 1, agg_str_lower.size() - pos - 2));

    if (agg_name == "count" && (col_name.empty() || col_name == "1" || col_name == "*")) {
      return count();
    }

    double val = HUGE_VAL;
    auto comma_pos = col_name.find(',');
    if (comma_pos != std::string::npos) {
      auto val_str = boost::trim_copy(
          col_name.substr(comma_pos + 1, col_name.size() - comma_pos - 1));
      char* end = nullptr;
      val = std::strtod(val_str.c_str(), &end);
      // Require value string to be fully interpreted to avoid silent errors like
      // 1..1 interpreted as 1.
      if (val == HUGE_VAL || end == val_str.c_str() ||
          end != (val_str.c_str() + val_str.size())) {
        throw InvalidQueryError()
            << "Cannot parse aggregate parameter (decimal expected): " << val_str;
      }
      col_name = boost::trim_copy(col_name.substr(0, comma_pos));
    }
    return ref(col_name).agg(agg_name, val);
  }

  throw InvalidQueryError() << "Cannot parse aggregate string: '" << agg_str << "'";
}

std::vector<BuilderExpr> BuilderNode::parseAggString(
    const std::vector<std::string>& aggs) const {
  std::vector<BuilderExpr> res;
  res.reserve(aggs.size());
  for (auto& agg_str : aggs) {
    res.emplace_back(parseAggString(agg_str));
  }
  return res;
}

BuilderNode BuilderNode::agg(int group_key, const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_key), std::vector<BuilderExpr>());
  }
  return agg(ref(group_key), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(int group_key,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(int group_key, const std::vector<std::string>& aggs) const {
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(int group_key, BuilderExpr agg_expr) const {
  return agg(ref(group_key), agg_expr);
}

BuilderNode BuilderNode::agg(int group_key, const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_key), aggs);
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::string& agg_str) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(agg_str));
  }
  if (agg_str.empty()) {
    return agg(ref(group_key), std::vector<BuilderExpr>());
  }
  return agg(ref(group_key), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             std::initializer_list<std::string> aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(aggs));
  }
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::vector<std::string>& aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(aggs));
  }
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::string& group_key, BuilderExpr agg_expr) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), agg_expr);
  }
  return agg(ref(group_key), agg_expr);
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::vector<BuilderExpr>& aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), aggs);
  }
  return agg(ref(group_key), aggs);
}

BuilderNode BuilderNode::agg(BuilderExpr group_key, const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(std::vector<BuilderExpr>({group_key}), std::vector<BuilderExpr>());
  }
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             std::initializer_list<std::string> aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             const std::vector<std::string>& aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key, BuilderExpr agg_expr) const {
  return agg(std::vector<BuilderExpr>({group_key}), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}
BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(std::vector<BuilderExpr>(group_keys), std::vector<BuilderExpr>());
  }
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(std::vector<BuilderExpr>(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(group_keys, std::vector<BuilderExpr>());
  }
  return agg(group_keys, parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(group_keys, parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(group_keys, parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(group_keys, std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  if (group_keys.empty() && aggs.empty()) {
    throw InvalidQueryError(
        "Empty aggregations are not allowed. At least one group key or aggregate is "
        "required.");
  }

  checkExprInput(group_keys, {node_.get()}, "aggregation");
  checkExprInput(aggs, {node_.get()}, "aggregation");

  std::vector<int> shuffle;
  std::vector<int> rev_shuffle(node_->size(), -1);
  shuffle.reserve(node_->size());
  for (auto& key : group_keys) {
    if (!key.expr_->is<ir::ColumnRef>()) {
      throw InvalidQueryError()
          << "Aggregation group key should be a column reference. Passed expression: "
          << key.expr_->toString();
    }
    auto col_idx = key.expr_->as<ir::ColumnRef>()->index();
    if (rev_shuffle[col_idx] == -1) {
      rev_shuffle[col_idx] = shuffle.size();
    }
    shuffle.push_back(col_idx);
  }

  for (auto& agg_expr : aggs) {
    if (!agg_expr.expr()->is<AggExpr>()) {
      throw InvalidQueryError() << "Non-aggregte expression is used as an aggregate: "
                                << agg_expr.expr()->toString();
    }
  }

  // Aggregate node requires all key columns to be first in the list
  // of input columns. Make additional projection to achieve that.
  // We also add a projection when aggregate over a scan because
  // such construction is not supported.
  if (!isIdentShuffle(shuffle) || node_->is<Scan>()) {
    for (size_t i = 0; i < rev_shuffle.size(); ++i) {
      if (rev_shuffle[i] == -1) {
        rev_shuffle[i] = shuffle.size();
        shuffle.push_back(i);
      }
    }

    auto base = proj(shuffle);
    std::vector<BuilderExpr> new_keys;
    new_keys.reserve(group_keys.size());
    for (int i = 0; i < (int)group_keys.size(); ++i) {
      if (group_keys[i].isAutoNamed()) {
        new_keys.emplace_back(base.ref(i));
      } else {
        new_keys.emplace_back(base.ref(i).name(group_keys[i].name()));
      }
    }
    return base.agg(new_keys, replaceInput(aggs, base.node_.get(), rev_shuffle));
  }

  auto all_exprs = group_keys;
  for (auto& agg : aggs) {
    all_exprs.emplace_back(agg);
  }
  auto agg_node = std::make_shared<Aggregate>(
      group_keys.size(), collectExprs(aggs), buildFieldNames(all_exprs), node_);
  return {builder_, agg_node};
}

namespace {

template <typename ColsType, typename... Ts>
std::vector<BuilderSortField> toSortFields(ColsType&& cols, Ts... args) {
  std::vector<BuilderSortField> res;
  for (auto& col : cols) {
    res.emplace_back(col, args...);
  }
  return res;
}

}  // namespace

BuilderNode BuilderNode::sort(int col_idx,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_idx, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(int col_idx,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_idx, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<int> col_indexes,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<int> col_indexes,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<int>& col_indexes,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<int>& col_indexes,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::string& col_name,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_name, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(const std::string& col_name,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_name, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<std::string> col_names,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<std::string> col_names,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<std::string>& col_names,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<std::string>& col_names,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(BuilderExpr col_ref,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_ref, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(BuilderExpr col_ref,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_ref, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<BuilderExpr> col_refs,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<BuilderExpr> col_refs,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderExpr>& col_refs,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderExpr>& col_refs,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const BuilderSortField& field,
                              size_t limit,
                              size_t offset) const {
  return sort(std::vector<BuilderSortField>({field}), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderSortField>& fields,
                              size_t limit,
                              size_t offset) const {
  std::vector<SortField> collation;
  for (auto field : fields) {
    ExprPtr col_ref;
    if (field.hasColIdx()) {
      col_ref = ref(field.colIdx()).expr();
    } else if (field.hasColName()) {
      col_ref = ref(field.colName()).expr();
    } else {
      CHECK(field.hasExpr());
      col_ref = field.expr();
      if (field.expr()->as<ColumnRef>()->node() != node_.get()) {
        throw InvalidQueryError()
            << "Sort field refers non-input column: " << field.expr()->toString();
      }
    }
    if (col_ref->type()->isArray()) {
      throw InvalidQueryError() << "Cannot sort on array column: " << col_ref->toString();
    }
    collation.emplace_back(
        col_ref->as<ColumnRef>()->index(), field.dir(), field.nullsPosition());
  }
  // Sort over scan is not supported.
  auto base = node_;
  if (node_->is<Scan>()) {
    base = proj().node();
  }
  auto sort_node = std::make_shared<Sort>(std::move(collation), limit, offset, base);
  return {builder_, sort_node};
}

std::unique_ptr<QueryDag> BuilderNode::finalize() const {
  // Scan and join nodes are not supposed to be a root of a query DAG.
  // Add a projection in such cases.
  if (auto scan = node_->as<Scan>()) {
    std::vector<int> cols;
    cols.reserve(scan->size());
    for (int i = 0; i < (int)node_->size(); ++i) {
      if (!scan->isVirtualCol(i)) {
        cols.push_back(i);
      }
    }
    return proj(cols).finalize();
  }

  return std::make_unique<QueryDag>(builder_.config_, node_);
}

QueryBuilder::QueryBuilder(Context& ctx,
                           SchemaProviderPtr schema_provider,
                           ConfigPtr config)
    : ctx_(ctx), schema_provider_(schema_provider), config_(config) {}

BuilderNode QueryBuilder::scan(const std::string& table_name) const {
  auto db_ids = schema_provider_->listDatabases();
  TableInfoPtr found_table = nullptr;
  for (auto db_id : db_ids) {
    auto table_info = schema_provider_->getTableInfo(db_id, table_name);
    if (table_info) {
      if (found_table) {
        throw InvalidQueryError() << "Ambiguous table name: " << table_name;
      }
      found_table = table_info;
    }
  }
  if (!found_table) {
    throw InvalidQueryError() << "Unknown table: " << table_name;
  }
  return scan(found_table);
}

BuilderNode QueryBuilder::scan(int db_id, const std::string& table_name) const {
  auto table_info = schema_provider_->getTableInfo(db_id, table_name);
  if (!table_info) {
    throw InvalidQueryError() << "Unknown table: " << table_name << " (db_id=" << db_id
                              << ")";
  }
  return scan(table_info);
}

BuilderNode QueryBuilder::scan(int db_id, int table_id) const {
  auto table_info = schema_provider_->getTableInfo(db_id, table_id);
  if (!table_info) {
    throw InvalidQueryError() << "Unknown table reference: db_id=" << db_id
                              << " table_id=" << table_id;
  }
  return scan(table_info);
}

BuilderNode QueryBuilder::scan(const TableRef& table_ref) const {
  return scan(table_ref.db_id, table_ref.table_id);
}

BuilderNode QueryBuilder::scan(TableInfoPtr table_info) const {
  auto scan =
      std::make_shared<Scan>(table_info, schema_provider_->listColumns(*table_info));
  return {*this, scan};
}

BuilderExpr QueryBuilder::count() const {
  auto count_type =
      config_->exec.group_by.bigint_count ? ctx_.int64(false) : ctx_.int32(false);
  auto agg = makeExpr<AggExpr>(count_type, AggType::kCount, nullptr, false, nullptr);
  return {*this, agg, "count", true};
}

BuilderExpr QueryBuilder::cst(int val) const {
  return cst(static_cast<int64_t>(val));
}

BuilderExpr QueryBuilder::cst(int val, const Type* type) const {
  return cst(static_cast<int64_t>(val), type);
}

BuilderExpr QueryBuilder::cst(int val, const std::string& type) const {
  return cst(static_cast<int64_t>(val), type);
}

BuilderExpr QueryBuilder::cst(int64_t val) const {
  return cst(val, ctx_.int64(false));
}

BuilderExpr QueryBuilder::cst(int64_t val, const Type* type) const {
  if (!type->isNumber() && !type->isDateTime() && !type->isInterval() &&
      !type->isBoolean()) {
    throw InvalidQueryError()
        << "Cannot create a literal from an integer value for type: " << type->toString();
  }
  if (type->isDate() && type->as<DateType>()->unit() == TimeUnit::kDay) {
    throw InvalidQueryError("Literals of date type with DAY time unit are not allowed.");
  }
  auto cst_expr = Constant::make(type, val);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cst(int64_t val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(double val) const {
  return cst(val, ctx_.fp64());
}

BuilderExpr QueryBuilder::cst(double val, const Type* type) const {
  Datum d;
  if (type->isFp32()) {
    d.floatval = static_cast<float>(val);
  } else if (type->isFp64()) {
    d.doubleval = val;
  } else if (type->isDecimal()) {
    d.bigintval =
        static_cast<int64_t>(val * exp_to_scale(type->as<DecimalType>()->scale()));
  } else {
    throw InvalidQueryError() << "Cannot create a literal from a double value for type: "
                              << type->toString();
  }
  auto cst_expr = std::make_shared<Constant>(type, false, d);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cst(double val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::string& val) const {
  return cst(val, ctx_.text());
}

BuilderExpr QueryBuilder::cst(const std::string& val, const Type* type) const {
  if (type->isDate() && type->as<DateType>()->unit() == TimeUnit::kDay) {
    throw InvalidQueryError("Literals of date type with DAY time unit are not allowed.");
  }
  if (type->isArray()) {
    throw InvalidQueryError("Cannot parse string to an array literal.");
  }
  try {
    Datum d;
    d.stringval = new std::string(val);
    auto cst_expr = std::make_shared<Constant>(ctx_.text(), false, d);
    return {*this, cst_expr->cast(type)};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError(e.what());
  }
}

BuilderExpr QueryBuilder::cst(const std::string& val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cstNoScale(int64_t val, const Type* type) const {
  if (!type->isDecimal()) {
    throw InvalidQueryError()
        << "Only decimal types are allowed for QueryBuilder::cstNoScale. Provided: "
        << type->toString();
  }
  Datum d;
  d.bigintval = val;
  auto cst_expr = std::make_shared<Constant>(type, false, d);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cstNoScale(int64_t val, const std::string& type) const {
  return cstNoScale(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::trueCst() const {
  Datum d;
  d.boolval = true;
  auto cst_expr = std::make_shared<Constant>(ctx_.boolean(false), false, d);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::falseCst() const {
  Datum d;
  d.boolval = true;
  auto cst_expr = std::make_shared<Constant>(ctx_.boolean(false), false, d);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::nullCst() const {
  return nullCst(ctx_.null());
}

BuilderExpr QueryBuilder::nullCst(const Type* type) const {
  auto cst_expr = std::make_shared<Constant>(type, true, Datum{});
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::nullCst(const std::string& type) const {
  return nullCst(ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int>& vals) const {
  return cst(std::vector<int>(vals));
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int> vals, const Type* type) const {
  return cst(std::vector<int>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int> vals,
                              const std::string& type) const {
  return cst(std::vector<int>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals) const {
  return cst(std::vector<double>(vals));
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals,
                              const Type* type) const {
  return cst(std::vector<double>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals,
                              const std::string& type) const {
  return cst(std::vector<double>(vals), type);
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals) const {
  return cst(vals, ctx_.arrayVarLen(ctx_.int32()));
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals, const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals) const {
  return cst(vals, ctx_.arrayVarLen(ctx_.fp64()));
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals, const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::vector<std::string>& vals,
                              const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {*this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<std::string>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

}  // namespace hdk::ir
