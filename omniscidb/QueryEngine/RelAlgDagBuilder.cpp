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

#include "IR/ExprRewriter.h"

#include "CalciteDeserializerUtils.h"
#include "DateTimePlusRewrite.h"
#include "DateTimeTranslator.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "ExprByPredicateCollector.h"
#include "ExpressionRewrite.h"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "JsonAccessors.h"
#include "RelAlgDagBuilder.h"
#include "RelAlgOptimizer.h"
#include "ResultSetRegistry/ResultSetRegistry.h"
#include "ScalarExprVisitor.h"
#include "Shared/sqldefs.h"

#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <boost/functional/hash.hpp>

#include <string>
#include <unordered_set>

using namespace std::literals;

namespace std {
template <>
struct hash<std::pair<const hdk::ir::Node*, int>> {
  size_t operator()(const std::pair<const hdk::ir::Node*, int>& input_col) const {
    auto ptr_val = reinterpret_cast<const int64_t*>(&input_col.first);
    return static_cast<int64_t>(*ptr_val) ^ input_col.second;
  }
};
}  // namespace std

namespace {}  // namespace

namespace {

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

std::string json_node_to_string(const rapidjson::Value& node) noexcept {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  node.Accept(writer);
  return buffer.GetString();
}

hdk::ir::ExprPtr parse_expr(const rapidjson::Value& expr,
                            int db_id,
                            SchemaProviderPtr schema_provider,
                            RelAlgDagBuilder& root_dag_builder,
                            const hdk::ir::ExprPtrVector& ra_output);

hdk::ir::TimeUnit precisionToTimeUnit(int precision) {
  switch (precision) {
    case 0:
      return hdk::ir::TimeUnit::kSecond;
    case 3:
      return hdk::ir::TimeUnit::kMilli;
    case 6:
      return hdk::ir::TimeUnit::kMicro;
    case 9:
      return hdk::ir::TimeUnit::kNano;
    default:
      throw std::runtime_error("Unsupported datetime precision: " +
                               std::to_string(precision));
  }
}

const hdk::ir::Type* buildType(hdk::ir::Context& ctx,
                               const std::string& type_name,
                               bool nullable,
                               int precision,
                               int scale) {
  if (type_name == std::string("BIGINT")) {
    return ctx.int64(nullable);
  }
  if (type_name == std::string("INTEGER")) {
    return ctx.int32(nullable);
  }
  if (type_name == std::string("TINYINT")) {
    return ctx.int8(nullable);
  }
  if (type_name == std::string("SMALLINT")) {
    return ctx.int16(nullable);
  }
  if (type_name == std::string("FLOAT")) {
    return ctx.fp32(nullable);
  }
  if (type_name == std::string("REAL")) {
    return ctx.fp32(nullable);
  }
  if (type_name == std::string("DOUBLE")) {
    return ctx.fp64(nullable);
  }
  if (type_name == std::string("DECIMAL")) {
    return ctx.decimal64(precision, scale, nullable);
  }
  if (type_name == std::string("CHAR") || type_name == std::string("VARCHAR")) {
    return ctx.text(nullable);
  }
  if (type_name == std::string("BOOLEAN")) {
    return ctx.boolean(nullable);
  }
  if (type_name == std::string("TIMESTAMP")) {
    return ctx.timestamp(precisionToTimeUnit(precision), nullable);
  }
  if (type_name == std::string("DATE")) {
    return ctx.date64(hdk::ir::TimeUnit::kSecond, nullable);
  }
  if (type_name == std::string("TIME")) {
    return ctx.time64(precisionToTimeUnit(precision), nullable);
  }
  if (type_name == std::string("NULL")) {
    return ctx.null();
  }
  if (type_name == std::string("ARRAY")) {
    return ctx.arrayVarLen(ctx.null(), 4, nullable);
  }
  if (type_name == std::string("INTERVAL_DAY") ||
      type_name == std::string("INTERVAL_HOUR") ||
      type_name == std::string("INTERVAL_MINUTE") ||
      type_name == std::string("INTERVAL_SECOND")) {
    return ctx.interval64(hdk::ir::TimeUnit::kMilli, nullable);
  }
  if (type_name == std::string("INTERVAL_MONTH") ||
      type_name == std::string("INTERVAL_YEAR")) {
    return ctx.interval64(hdk::ir::TimeUnit::kMonth, nullable);
  }
  if (type_name == std::string("TEXT")) {
    return ctx.text(nullable);
  }

  throw std::runtime_error("Unsupported type: " + type_name);
}

const hdk::ir::Type* parseType(const rapidjson::Value& type_obj) {
  if (type_obj.IsArray()) {
    throw hdk::ir::QueryNotSupported("Composite types are not currently supported.");
  }
  CHECK(type_obj.IsObject() && type_obj.MemberCount() >= 2)
      << json_node_to_string(type_obj);
  auto type_name = json_str(field(type_obj, "type"));
  auto nullable = json_bool(field(type_obj, "nullable"));
  auto precision_it = type_obj.FindMember("precision");
  int precision =
      precision_it != type_obj.MemberEnd() ? json_i64(precision_it->value) : 0;
  auto scale_it = type_obj.FindMember("scale");
  int scale = scale_it != type_obj.MemberEnd() ? json_i64(scale_it->value) : 0;

  return buildType(hdk::ir::Context::defaultCtx(), type_name, nullable, precision, scale);
}

hdk::ir::WindowFunctionKind parse_window_function_kind(const std::string& name) {
  if (name == "ROW_NUMBER") {
    return hdk::ir::WindowFunctionKind::RowNumber;
  }
  if (name == "RANK") {
    return hdk::ir::WindowFunctionKind::Rank;
  }
  if (name == "DENSE_RANK") {
    return hdk::ir::WindowFunctionKind::DenseRank;
  }
  if (name == "PERCENT_RANK") {
    return hdk::ir::WindowFunctionKind::PercentRank;
  }
  if (name == "CUME_DIST") {
    return hdk::ir::WindowFunctionKind::CumeDist;
  }
  if (name == "NTILE") {
    return hdk::ir::WindowFunctionKind::NTile;
  }
  if (name == "LAG") {
    return hdk::ir::WindowFunctionKind::Lag;
  }
  if (name == "LEAD") {
    return hdk::ir::WindowFunctionKind::Lead;
  }
  if (name == "FIRST_VALUE") {
    return hdk::ir::WindowFunctionKind::FirstValue;
  }
  if (name == "LAST_VALUE") {
    return hdk::ir::WindowFunctionKind::LastValue;
  }
  if (name == "AVG") {
    return hdk::ir::WindowFunctionKind::Avg;
  }
  if (name == "MIN") {
    return hdk::ir::WindowFunctionKind::Min;
  }
  if (name == "MAX") {
    return hdk::ir::WindowFunctionKind::Max;
  }
  if (name == "SUM") {
    return hdk::ir::WindowFunctionKind::Sum;
  }
  if (name == "COUNT") {
    return hdk::ir::WindowFunctionKind::Count;
  }
  if (name == "$SUM0") {
    return hdk::ir::WindowFunctionKind::SumInternal;
  }
  throw std::runtime_error("Unsupported window function: " + name);
}

hdk::ir::SortDirection parse_sort_direction(const rapidjson::Value& collation) {
  return json_str(field(collation, "direction")) == std::string("DESCENDING")
             ? hdk::ir::SortDirection::Descending
             : hdk::ir::SortDirection::Ascending;
}

hdk::ir::NullSortedPosition parse_nulls_position(const rapidjson::Value& collation) {
  return json_str(field(collation, "nulls")) == std::string("FIRST")
             ? hdk::ir::NullSortedPosition::First
             : hdk::ir::NullSortedPosition::Last;
}

std::vector<hdk::ir::OrderEntry> parseWindowOrderCollation(const rapidjson::Value& arr) {
  std::vector<hdk::ir::OrderEntry> collation;
  size_t field_idx = 0;
  for (auto it = arr.Begin(); it != arr.End(); ++it, ++field_idx) {
    const auto sort_dir = parse_sort_direction(*it);
    const auto null_pos = parse_nulls_position(*it);
    collation.emplace_back(field_idx,
                           sort_dir == hdk::ir::SortDirection::Descending,
                           null_pos == hdk::ir::NullSortedPosition::First);
  }
  return collation;
}

struct WindowBound {
  bool unbounded;
  bool preceding;
  bool following;
  bool is_current_row;
  hdk::ir::ExprPtr offset;
  int order_key;
};

WindowBound parse_window_bound(const rapidjson::Value& window_bound_obj,
                               int db_id,
                               SchemaProviderPtr schema_provider,
                               RelAlgDagBuilder& root_dag_builder,
                               const hdk::ir::ExprPtrVector& ra_output) {
  CHECK(window_bound_obj.IsObject());
  WindowBound window_bound;
  window_bound.unbounded = json_bool(field(window_bound_obj, "unbounded"));
  window_bound.preceding = json_bool(field(window_bound_obj, "preceding"));
  window_bound.following = json_bool(field(window_bound_obj, "following"));
  window_bound.is_current_row = json_bool(field(window_bound_obj, "is_current_row"));
  const auto& offset_field = field(window_bound_obj, "offset");
  if (offset_field.IsObject()) {
    window_bound.offset =
        parse_expr(offset_field, db_id, schema_provider, root_dag_builder, ra_output);
  } else {
    CHECK(offset_field.IsNull());
  }
  window_bound.order_key = json_i64(field(window_bound_obj, "order_key"));
  return window_bound;
}

hdk::ir::ExprPtr parse_subquery_expr(const rapidjson::Value& expr,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(0));
  const auto& subquery_ast = field(expr, "subquery");

  RelAlgDagBuilder subquery_dag(root_dag_builder, subquery_ast, db_id, schema_provider);
  auto node = subquery_dag.getRootNodeShPtr();
  auto subquery =
      hdk::ir::makeExpr<hdk::ir::ScalarSubquery>(getColumnType(node.get(), 0), node);
  root_dag_builder.registerSubquery(subquery);
  return subquery;
}

std::vector<std::string> strings_from_json_array(
    const rapidjson::Value& json_str_arr) noexcept {
  CHECK(json_str_arr.IsArray());
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End();
       ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
}

std::vector<size_t> indices_from_json_array(
    const rapidjson::Value& json_idx_arr) noexcept {
  CHECK(json_idx_arr.IsArray());
  std::vector<size_t> indices;
  for (auto json_idx_arr_it = json_idx_arr.Begin(); json_idx_arr_it != json_idx_arr.End();
       ++json_idx_arr_it) {
    CHECK(json_idx_arr_it->IsInt());
    CHECK_GE(json_idx_arr_it->GetInt(), 0);
    indices.emplace_back(json_idx_arr_it->GetInt());
  }
  return indices;
}

hdk::ir::ExprPtr parseInput(const rapidjson::Value& expr,
                            const hdk::ir::ExprPtrVector& ra_output) {
  const auto& input = field(expr, "input");
  CHECK_LT(json_i64(input), ra_output.size());
  return ra_output[json_i64(input)];
}

hdk::ir::ExprPtr parseLiteral(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type_name = json_str(field(expr, "type"));
  const auto target_type_name = json_str(field(expr, "target_type"));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));

  auto& ctx = hdk::ir::Context::defaultCtx();
  auto lit_type = buildType(ctx, type_name, false, precision, scale);
  auto target_type = buildType(ctx, target_type_name, false, type_precision, type_scale);

  if (literal.IsNull()) {
    return hdk::ir::makeExpr<hdk::ir::Constant>(
        target_type->withNullable(true), true, Datum{0});
  }

  switch (lit_type->id()) {
    case hdk::ir::Type::kInteger: {
      Datum d;
      d.bigintval = json_i64(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(lit_type, false, d);
    }
    case hdk::ir::Type::kDecimal: {
      int64_t val = json_i64(literal);
      if (target_type->isFloatingPoint() && !scale) {
        return make_fp_constant(val, target_type);
      }
      auto lit_expr = scale ? Analyzer::analyzeFixedPtValue(val, scale, precision)
                            : Analyzer::analyzeIntValue(val);
      return lit_type->equal(target_type) ? lit_expr : lit_expr->cast(target_type);
    }
    case hdk::ir::Type::kText: {
      return Analyzer::analyzeStringValue(json_str(literal));
    }
    case hdk::ir::Type::kBoolean: {
      Datum d;
      d.boolval = json_bool(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(lit_type, false, d);
    }
    case hdk::ir::Type::kFloatingPoint: {
      Datum d;
      if (literal.IsDouble()) {
        d.doubleval = json_double(literal);
      } else if (literal.IsInt64()) {
        d.doubleval = static_cast<double>(literal.GetInt64());
      } else if (literal.IsUint64()) {
        d.doubleval = static_cast<double>(literal.GetUint64());
      } else {
        UNREACHABLE() << "Unhandled type: " << literal.GetType();
      }
      auto lit_expr = hdk::ir::makeExpr<hdk::ir::Constant>(ctx.fp64(false), false, d);
      return lit_type->equal(target_type) ? lit_expr : lit_expr->cast(target_type);
    }
    case hdk::ir::Type::kInterval: {
      Datum d;
      d.bigintval = json_i64(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(lit_type, false, d);
    }
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp: {
      Datum d;
      d.bigintval = json_i64(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(lit_type, false, d);
    }
    case hdk::ir::Type::kDate: {
      Datum d;
      d.bigintval = json_i64(literal) * 24 * 3600;
      return hdk::ir::makeExpr<hdk::ir::Constant>(lit_type, false, d);
    }
    case hdk::ir::Type::kNull: {
      if (target_type->isArray()) {
        hdk::ir::ExprPtrVector args;
        // defaulting to valid sub-type for convenience
        target_type =
            target_type->as<hdk::ir::ArrayBaseType>()->withElemType(ctx.boolean());
        return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(target_type, args, true);
      }
      return hdk::ir::makeExpr<hdk::ir::Constant>(target_type, true, Datum{0});
    }
    default: {
      LOG(FATAL) << "Unexpected literal type " << lit_type->toString();
    }
  }
  return nullptr;
}

hdk::ir::ExprPtr parse_case_expr(const rapidjson::Value& expr,
                                 int db_id,
                                 SchemaProviderPtr schema_provider,
                                 RelAlgDagBuilder& root_dag_builder,
                                 const hdk::ir::ExprPtrVector& ra_output) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  hdk::ir::ExprPtr else_expr;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr =
        parse_expr(*operands_it++, db_id, schema_provider, root_dag_builder, ra_output);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr =
        parse_expr(*operands_it++, db_id, schema_provider, root_dag_builder, ra_output);
    expr_list.emplace_back(std::move(when_expr), std::move(then_expr));
  }
  return Analyzer::normalizeCaseExpr(expr_list, else_expr, nullptr);
}

hdk::ir::ExprPtrVector parseExprArray(const rapidjson::Value& arr,
                                      int db_id,
                                      SchemaProviderPtr schema_provider,
                                      RelAlgDagBuilder& root_dag_builder,
                                      const hdk::ir::ExprPtrVector& ra_output) {
  hdk::ir::ExprPtrVector exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(
        parse_expr(*it, db_id, schema_provider, root_dag_builder, ra_output));
  }
  return exprs;
}

hdk::ir::ExprPtrVector parseWindowOrderExprs(const rapidjson::Value& arr,
                                             int db_id,
                                             SchemaProviderPtr schema_provider,
                                             RelAlgDagBuilder& root_dag_builder,
                                             const hdk::ir::ExprPtrVector& ra_output) {
  hdk::ir::ExprPtrVector exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(parse_expr(
        field(*it, "field"), db_id, schema_provider, root_dag_builder, ra_output));
  }
  return exprs;
}

hdk::ir::ExprPtr makeUOper(hdk::ir::OpType op,
                           hdk::ir::ExprPtr arg,
                           const hdk::ir::Type* type) {
  auto& ctx = type->ctx();
  switch (op) {
    case hdk::ir::OpType::kCast: {
      CHECK(!type->isNull());
      const auto& arg_type = arg->type();
      if ((arg_type->isString() || arg_type->isExtDictionary()) &&
          (type->isString() || type->isExtDictionary())) {
        return arg;
      }
      if (type->isDateTime() || arg_type->isString() || arg_type->isExtDictionary()) {
        // TODO(alex): check and unify with the rest of the cases
        // Do not propogate encoding on small dates
        if (type->isDate() &&
            type->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay) {
          return arg->cast(ctx.time64(hdk::ir::TimeUnit::kSecond));
        }
        return arg->cast(type);
      }
      if (!(arg_type->isString() || arg_type->isExtDictionary()) &&
          (type->isString() || type->isExtDictionary())) {
        return arg->cast(type);
      }
      return std::make_shared<hdk::ir::UOper>(type, false, op, arg);
    }
    case hdk::ir::OpType::kNot:
      return std::make_shared<hdk::ir::UOper>(
          ctx.boolean(arg->type()->nullable()), op, arg);
    case hdk::ir::OpType::kBwNot:
      if (!arg->type()->isInteger()) {
        throw std::runtime_error(
            "Only integer expressions are allowed for BW_NOT operation.");
      }
      return std::make_shared<hdk::ir::UOper>(arg->type(), op, arg);
    case hdk::ir::OpType::kIsNull:
      return std::make_shared<hdk::ir::UOper>(ctx.boolean(false), op, arg);
    case hdk::ir::OpType::kMinus: {
      return std::make_shared<hdk::ir::UOper>(
          arg->type(), false, hdk::ir::OpType::kUMinus, arg);
    }
    case hdk::ir::OpType::kUnnest: {
      const auto& arg_type = arg->type();
      CHECK(arg_type->isArray());
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          arg_type->as<hdk::ir::ArrayBaseType>()->elemType(),
          false,
          hdk::ir::OpType::kUnnest,
          arg);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

hdk::ir::ExprPtr maybeMakeDateExpr(hdk::ir::OpType op,
                                   const hdk::ir::ExprPtrVector& operands,
                                   const hdk::ir::Type* type) {
  if (op != hdk::ir::OpType::kPlus && op != hdk::ir::OpType::kMinus) {
    return nullptr;
  }
  if (operands.size() != 2) {
    return nullptr;
  }

  auto& lhs = operands[0];
  auto& rhs = operands[1];
  auto lhs_type = lhs->type();
  auto rhs_type = rhs->type();
  if (!lhs_type->isTimestamp() && !lhs_type->isDate()) {
    if (lhs_type->isTime()) {
      throw std::runtime_error("DateTime addition/subtraction not supported for TIME.");
    }
    return nullptr;
  }
  if (rhs_type->isTimestamp() || rhs_type->isDate()) {
    if (lhs_type->as<hdk::ir::DateTimeBaseType>()->unit() > hdk::ir::TimeUnit::kSecond ||
        rhs_type->as<hdk::ir::DateTimeBaseType>()->unit() > hdk::ir::TimeUnit::kSecond) {
      throw std::runtime_error(
          "High Precision timestamps are not supported for TIMESTAMPDIFF operation. "
          "Use "
          "DATEDIFF.");
    }
    auto bigint_type = type->ctx().int64(false);

    CHECK(op == hdk::ir::OpType::kMinus);
    CHECK(type->isInterval());
    auto res_unit = type->as<hdk::ir::IntervalType>()->unit();
    if (res_unit == hdk::ir::TimeUnit::kMilli) {
      auto result = hdk::ir::makeExpr<hdk::ir::DateDiffExpr>(
          bigint_type, hdk::ir::DateTruncField::kSecond, rhs, lhs);
      // multiply 1000 to result since expected result should be in millisecond precision.
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bigint_type,
          hdk::ir::OpType::kMul,
          hdk::ir::Qualifier::kOne,
          result,
          hdk::ir::Constant::make(bigint_type, 1000));

    } else if (res_unit == hdk::ir::TimeUnit::kMilli) {
      return hdk::ir::makeExpr<hdk::ir::DateDiffExpr>(
          bigint_type, hdk::ir::DateTruncField::kMonth, rhs, lhs);

    } else {
      throw std::runtime_error("Unexpected DATEDIFF result type" + type->toString());
    }
  }
  if (op == hdk::ir::OpType::kPlus) {
    auto dt_plus =
        hdk::ir::makeExpr<hdk::ir::FunctionOper>(lhs_type, "DATETIME_PLUS", operands);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
  }
  const auto interval = fold_expr(rhs.get());
  auto interval_type = interval->type()->as<hdk::ir::IntervalType>();
  CHECK(interval_type);
  auto bigint_type = type->ctx().int64(false);
  const auto interval_lit = interval->as<hdk::ir::Constant>();
  if (interval_type->unit() == hdk::ir::TimeUnit::kMilli) {
    hdk::ir::ExprPtr interval_sec;
    if (interval_lit) {
      interval_sec = hdk::ir::Constant::make(
          bigint_type,
          (op == hdk::ir::OpType::kMinus ? -interval_lit->value().bigintval
                                         : interval_lit->value().bigintval) /
              1000);
    } else {
      interval_sec =
          hdk::ir::makeExpr<hdk::ir::BinOper>(bigint_type,
                                              hdk::ir::OpType::kDiv,
                                              hdk::ir::Qualifier::kOne,
                                              interval,
                                              hdk::ir::Constant::make(bigint_type, 1000));
      if (op == hdk::ir::OpType::kMinus) {
        interval_sec = std::make_shared<hdk::ir::UOper>(
            bigint_type, false, hdk::ir::OpType::kUMinus, interval_sec);
      }
    }
    return hdk::ir::makeExpr<hdk::ir::DateAddExpr>(
        lhs_type, hdk::ir::DateAddField::kSecond, interval_sec, lhs);
  }
  CHECK(interval_type->unit() == hdk::ir::TimeUnit::kMonth);
  const auto interval_months =
      op == hdk::ir::OpType::kMinus
          ? std::make_shared<hdk::ir::UOper>(
                bigint_type, false, hdk::ir::OpType::kUMinus, interval)
          : interval;
  return hdk::ir::makeExpr<hdk::ir::DateAddExpr>(
      lhs_type, hdk::ir::DateAddField::kMonth, interval_months, lhs);
}

std::pair<hdk::ir::ExprPtr, hdk::ir::Qualifier> getQuantifiedBinOperRhs(
    const hdk::ir::ExprPtr& expr,
    hdk::ir::ExprPtr orig_expr) {
  if (auto fn_oper = dynamic_cast<const hdk::ir::FunctionOper*>(expr.get())) {
    auto& fn_name = fn_oper->name();
    if (fn_name == "PG_ANY" || fn_name == "PG_ALL") {
      CHECK_EQ(fn_oper->arity(), (size_t)1);
      return std::make_pair(
          fn_oper->argShared(0),
          (fn_name == "PG_ANY") ? hdk::ir::Qualifier::kAny : hdk::ir::Qualifier::kAll);
    }
  } else if (auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr.get())) {
    if (uoper->isCast()) {
      return getQuantifiedBinOperRhs(uoper->operandShared(), orig_expr);
    }
  }

  return std::make_pair(orig_expr, hdk::ir::Qualifier::kOne);
}

std::pair<hdk::ir::ExprPtr, hdk::ir::Qualifier> getQuantifiedBinOperRhs(
    const hdk::ir::ExprPtr& expr) {
  return getQuantifiedBinOperRhs(expr, expr);
}

bool supportedLowerBound(const WindowBound& window_bound) {
  return window_bound.unbounded && window_bound.preceding && !window_bound.following &&
         !window_bound.is_current_row && !window_bound.offset &&
         window_bound.order_key == 0;
}

bool supportedUpperBound(const WindowBound& window_bound,
                         hdk::ir::WindowFunctionKind kind,
                         const hdk::ir::ExprPtrVector& order_keys) {
  const bool to_current_row = !window_bound.unbounded && !window_bound.preceding &&
                              !window_bound.following && window_bound.is_current_row &&
                              !window_bound.offset && window_bound.order_key == 1;
  switch (kind) {
    case hdk::ir::WindowFunctionKind::RowNumber:
    case hdk::ir::WindowFunctionKind::Rank:
    case hdk::ir::WindowFunctionKind::DenseRank:
    case hdk::ir::WindowFunctionKind::CumeDist: {
      return to_current_row;
    }
    default: {
      return order_keys.empty()
                 ? (window_bound.unbounded && !window_bound.preceding &&
                    window_bound.following && !window_bound.is_current_row &&
                    !window_bound.offset && window_bound.order_key == 2)
                 : to_current_row;
    }
  }
}

hdk::ir::ExprPtr parseWindowFunction(const rapidjson::Value& json_expr,
                                     const std::string& op_name,
                                     const hdk::ir::ExprPtrVector& operands,
                                     const hdk::ir::Type* type,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder,
                                     const hdk::ir::ExprPtrVector& ra_output) {
  const auto& partition_keys_arr = field(json_expr, "partition_keys");
  auto partition_keys = parseExprArray(
      partition_keys_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto& order_keys_arr = field(json_expr, "order_keys");
  auto order_keys = parseWindowOrderExprs(
      order_keys_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto collation = parseWindowOrderCollation(order_keys_arr);
  const auto kind = parse_window_function_kind(op_name);
  // Adjust type for SUM window function.
  if (kind == hdk::ir::WindowFunctionKind::SumInternal && type->isInteger()) {
    type = type->ctx().int64(type->nullable());
  }
  const auto lower_bound = parse_window_bound(field(json_expr, "lower_bound"),
                                              db_id,
                                              schema_provider,
                                              root_dag_builder,
                                              ra_output);
  const auto upper_bound = parse_window_bound(field(json_expr, "upper_bound"),
                                              db_id,
                                              schema_provider,
                                              root_dag_builder,
                                              ra_output);
  bool is_rows = json_bool(field(json_expr, "is_rows"));
  type = type->withNullable(true);

  if (!supportedLowerBound(lower_bound) ||
      !supportedUpperBound(upper_bound, kind, order_keys) ||
      ((kind == hdk::ir::WindowFunctionKind::RowNumber) != is_rows)) {
    throw std::runtime_error("Frame specification not supported");
  }

  if (window_function_is_value(kind)) {
    CHECK_GE(operands.size(), 1u);
    type = operands[0]->type();
  }

  return hdk::ir::makeExpr<hdk::ir::WindowFunction>(
      type, kind, operands, partition_keys, order_keys, collation);
}

hdk::ir::ExprPtr parseLike(const std::string& fn_name,
                           const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 2 || operands.size() == 3);
  auto& arg = operands[0];
  auto& like = operands[1];
  if (!dynamic_cast<const hdk::ir::Constant*>(like.get())) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  auto escape = (operands.size() == 3) ? operands[2] : nullptr;
  bool is_ilike = fn_name == "PG_ILIKE"sv;
  return Analyzer::getLikeExpr(arg, like, escape, is_ilike, false);
}

hdk::ir::ExprPtr parseRegexp(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 2 || operands.size() == 3);
  auto& arg = operands[0];
  auto& pattern = operands[1];
  if (!dynamic_cast<const hdk::ir::Constant*>(pattern.get())) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (operands.size() == 3) ? operands[2] : nullptr;
  return Analyzer::getRegexpExpr(arg, pattern, escape, false);
}

hdk::ir::ExprPtr parseLikely(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 1);
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(operands[0], 0.9375);
}

hdk::ir::ExprPtr parseUnlikely(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 1);
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(operands[0], 0.0625);
}

inline void validateDatetimeDatepartArgument(const hdk::ir::Constant* literal_expr) {
  if (!literal_expr || literal_expr->isNull()) {
    throw std::runtime_error("The 'DatePart' argument must be a not 'null' literal.");
  }
}

hdk::ir::ExprPtr parseExtract(const std::string& fn_name,
                              const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& from_expr = operands[1];
  if (fn_name == "PG_DATE_TRUNC"sv) {
    return DateTruncExpr::generate(from_expr, *timeunit_lit->value().stringval);
  } else {
    CHECK(fn_name == "PG_EXTRACT"sv);
    return ExtractExpr::generate(from_expr, *timeunit_lit->value().stringval);
  }
}

hdk::ir::ExprPtr parseDateadd(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(3));
  auto& ctx = operands[0]->type()->ctx();
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& number_units = operands[1];
  auto number_units_const = dynamic_cast<const hdk::ir::Constant*>(number_units.get());
  if (number_units_const && number_units_const->isNull()) {
    throw std::runtime_error("The 'Interval' argument literal must not be 'null'.");
  }
  auto cast_number_units = number_units->cast(ctx.int64());
  auto& datetime = operands[2];
  auto datetime_type = datetime->type();
  if (datetime_type->isTime()) {
    throw std::runtime_error("DateAdd operation not supported for TIME.");
  }
  auto field = to_dateadd_field(*timeunit_lit->value().stringval);
  auto unit = datetime_type->isTimestamp()
                  ? datetime_type->as<hdk::ir::TimestampType>()->unit()
                  : hdk::ir::TimeUnit::kSecond;
  return hdk::ir::makeExpr<hdk::ir::DateAddExpr>(
      ctx.timestamp(unit), field, cast_number_units, datetime);
}

hdk::ir::ExprPtr parseDatediff(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(3));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& start = operands[1];
  auto& end = operands[2];
  auto field = to_datediff_field(*timeunit_lit->value().stringval);
  return hdk::ir::makeExpr<hdk::ir::DateDiffExpr>(
      start->type()->ctx().int64(), field, start, end);
}

hdk::ir::ExprPtr parseDatepart(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& from_expr = operands[1];
  return ExtractExpr::generate(from_expr,
                               to_datepart_field(*timeunit_lit->value().stringval));
}

hdk::ir::ExprPtr parseLength(const std::string& fn_name,
                             const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto& str_arg = operands[0];
  return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(str_arg->decompress(),
                                                    fn_name == "CHAR_LENGTH"sv);
}

hdk::ir::ExprPtr parseKeyForString(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto arg_type = operands[0]->type();
  if (!arg_type->isExtDictionary()) {
    throw std::runtime_error("KEY_FOR_STRING expects a dictionary encoded text column.");
  }
  return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(operands[0]);
}

hdk::ir::ExprPtr parseWidthBucket(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(4));
  auto target_value = operands[0];
  auto lower_bound = operands[1];
  auto upper_bound = operands[2];
  auto partition_count = operands[3];
  if (!partition_count->type()->isInteger()) {
    throw std::runtime_error(
        "PARTITION_COUNT expression of width_bucket function expects an integer type.");
  }
  auto check_numeric_type =
      [](const std::string& col_name, const hdk::ir::Expr* expr, bool allow_null_type) {
        if (expr->type()->isNull()) {
          if (!allow_null_type) {
            throw std::runtime_error(
                col_name + " expression of width_bucket function expects non-null type.");
          }
          return;
        }
        if (!expr->type()->isNumber()) {
          throw std::runtime_error(
              col_name + " expression of width_bucket function expects a numeric type.");
        }
      };
  // target value may have null value
  check_numeric_type("TARGET_VALUE", target_value.get(), true);
  check_numeric_type("LOWER_BOUND", lower_bound.get(), false);
  check_numeric_type("UPPER_BOUND", upper_bound.get(), false);

  auto cast_to_double_if_necessary = [](hdk::ir::ExprPtr arg) {
    const auto& arg_type = arg->type();
    if (!arg_type->isFp64()) {
      return arg->cast(arg_type->ctx().fp64(arg_type->nullable()));
    }
    return arg;
  };
  target_value = cast_to_double_if_necessary(target_value);
  lower_bound = cast_to_double_if_necessary(lower_bound);
  upper_bound = cast_to_double_if_necessary(upper_bound);
  return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
      target_value, lower_bound, upper_bound, partition_count);
}

hdk::ir::ExprPtr parseSampleRatio(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto arg = operands[0];
  auto arg_type = operands[0]->type();
  if (!arg_type->isFp64()) {
    arg = arg->cast(arg_type->ctx().fp64(arg_type->nullable()));
  }
  return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(arg);
}

hdk::ir::ExprPtr parseCurrentUser() {
  std::string user{"SESSIONLESS_USER"};
  return Analyzer::getUserLiteral(user);
}

hdk::ir::ExprPtr parseLower(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  if (operands[0]->type()->isExtDictionary() ||
      dynamic_cast<const hdk::ir::Constant*>(operands[0].get())) {
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(operands[0]);
  }

  throw std::runtime_error(
      "LOWER expects a dictionary encoded text column or a literal.");
}

hdk::ir::ExprPtr parseCardinality(const std::string& fn_name,
                                  const hdk::ir::ExprPtrVector& operands,
                                  const hdk::ir::Type* type) {
  CHECK_EQ(operands.size(), size_t(1));
  auto& arg = operands[0];
  auto arg_type = arg->type();
  if (!arg_type->isArray()) {
    throw std::runtime_error(fn_name + " expects an array expression.");
  }
  auto elem_type = arg_type->as<hdk::ir::ArrayBaseType>()->elemType();
  if (elem_type->isArray()) {
    throw std::runtime_error(fn_name + " expects one-dimension array expression.");
  }
  auto array_size = arg_type->size();
  auto array_elem_size = elem_type->isString() ? 4 : elem_type->canonicalSize();

  if (array_size > 0) {
    if (array_elem_size <= 0) {
      throw std::runtime_error(fn_name + ": unexpected array element type.");
    }
    // Return cardinality of a fixed length array
    return hdk::ir::Constant::make(type, array_size / array_elem_size);
  }
  // Variable length array cardinality will be calculated at runtime
  return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(arg);
}

hdk::ir::ExprPtr parseItem(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& base = operands[0];
  auto& index = operands[1];
  CHECK(base->type()->isArray());
  return hdk::ir::makeExpr<hdk::ir::BinOper>(
      base->type()->as<hdk::ir::ArrayBaseType>()->elemType(),
      false,
      hdk::ir::OpType::kArrayAt,
      hdk::ir::Qualifier::kOne,
      base,
      index);
}

hdk::ir::ExprPtr parseCurrentDate(time_t now) {
  Datum datum;
  datum.bigintval = now - now % (24 * 60 * 60);  // Assumes 0 < now.
  return hdk::ir::makeExpr<hdk::ir::Constant>(
      hdk::ir::Context::defaultCtx().date64(hdk::ir::TimeUnit::kSecond, false),
      false,
      datum);
}

hdk::ir::ExprPtr parseCurrentTime(time_t now) {
  Datum datum;
  datum.bigintval = now % (24 * 60 * 60);  // Assumes 0 < now.
  return hdk::ir::makeExpr<hdk::ir::Constant>(
      hdk::ir::Context::defaultCtx().time64(hdk::ir::TimeUnit::kSecond, false),
      false,
      datum);
}

hdk::ir::ExprPtr parseCurrentTimestamp(time_t now) {
  Datum d;
  d.bigintval = now;
  return hdk::ir::makeExpr<hdk::ir::Constant>(
      hdk::ir::Context::defaultCtx().timestamp(hdk::ir::TimeUnit::kSecond, false),
      false,
      d,
      false);
}

hdk::ir::ExprPtr parseDatetime(const hdk::ir::ExprPtrVector& operands, time_t now) {
  CHECK_EQ(operands.size(), size_t(1));
  auto arg_lit = dynamic_cast<const hdk::ir::Constant*>(operands[0].get());
  const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
  if (!arg_lit || arg_lit->isNull()) {
    throw std::runtime_error(datetime_err);
  }
  CHECK(arg_lit->type()->isString() || arg_lit->type()->isExtDictionary());
  if (*arg_lit->value().stringval != "NOW"sv) {
    throw std::runtime_error(datetime_err);
  }
  return parseCurrentTimestamp(now);
}

hdk::ir::ExprPtr parseHPTLiteral(const hdk::ir::ExprPtrVector& operands,
                                 const hdk::ir::Type* type) {
  /* since calcite uses Avatica package called DateTimeUtils to parse timestamp strings.
     Therefore any string having fractional seconds more 3 places after the decimal
     (milliseconds) will get truncated to 3 decimal places, therefore we lose precision
     (us|ns). Issue: [BE-2461] Here we are hijacking literal cast to Timestamp(6|9) from
     calcite and translating them to generate our own casts.
  */
  CHECK_EQ(operands.size(), size_t(1));
  auto& arg = operands[0];
  auto arg_type = arg->type();
  if (!arg_type->isString() && !arg_type->isExtDictionary()) {
    throw std::runtime_error(
        "High precision timestamp cast argument must be a string. Input type is: " +
        arg_type->toString());
  } else if (!type->isTimestamp()) {
    throw std::runtime_error(
        "Cast target type should be high precision timestamp. Input type is: " +
        type->toString());
  } else if (type->as<hdk::ir::TimestampType>()->unit() < hdk::ir::TimeUnit::kMicro) {
    throw std::runtime_error(
        "Cast target type should be TIMESTAMP(6|9). Input type is: " + type->toString());
  }

  return arg->cast(type);
}

hdk::ir::ExprPtr parseAbs(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  auto& arg = operands[0];
  auto arg_type = arg->type();
  CHECK(arg_type->isNumber());
  auto zero = hdk::ir::Constant::make(arg_type, 0);
  auto lt_zero = Analyzer::normalizeOperExpr(
      hdk::ir::OpType::kLt, hdk::ir::Qualifier::kOne, arg, zero);
  auto uminus_operand =
      hdk::ir::makeExpr<hdk::ir::UOper>(arg_type, false, hdk::ir::OpType::kUMinus, arg);
  expr_list.emplace_back(lt_zero, uminus_operand);
  return Analyzer::normalizeCaseExpr(expr_list, arg, nullptr);
}

hdk::ir::ExprPtr parseSign(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  auto& arg = operands[0];
  auto arg_type = arg->type();
  CHECK(arg_type->isNumber());
  // For some reason, Rex based DAG checker marks SIGN as non-cacheable.
  // To duplicate this behavior with no Rex, non-cacheable zero constant
  // is used here.
  // TODO: revise this part in checker and probably remove this flag here.
  const auto zero = hdk::ir::Constant::make(arg_type, 0, false);
  auto bool_type = arg_type->ctx().boolean(arg_type->nullable());
  const auto lt_zero = hdk::ir::makeExpr<hdk::ir::BinOper>(
      bool_type, hdk::ir::OpType::kLt, hdk::ir::Qualifier::kOne, arg, zero);
  expr_list.emplace_back(lt_zero, hdk::ir::Constant::make(arg_type, -1));
  const auto eq_zero = hdk::ir::makeExpr<hdk::ir::BinOper>(
      bool_type, hdk::ir::OpType::kEq, hdk::ir::Qualifier::kOne, arg, zero);
  expr_list.emplace_back(eq_zero, hdk::ir::Constant::make(arg_type, 0));
  const auto gt_zero = hdk::ir::makeExpr<hdk::ir::BinOper>(
      bool_type, hdk::ir::OpType::kGt, hdk::ir::Qualifier::kOne, arg, zero);
  expr_list.emplace_back(gt_zero, hdk::ir::Constant::make(arg_type, 1));
  return Analyzer::normalizeCaseExpr(expr_list, nullptr, nullptr);
}

hdk::ir::ExprPtr parseRound(const std::string& fn_name,
                            const hdk::ir::ExprPtrVector& operands,
                            const hdk::ir::Type* type) {
  auto args = operands;

  if (args.size() == 1) {
    // push a 0 constant if 2nd operand is missing.
    // this needs to be done as calcite returns
    // only the 1st operand without defaulting the 2nd one
    // when the user did not specify the 2nd operand.
    Datum d;
    d.smallintval = 0;
    args.push_back(
        hdk::ir::makeExpr<hdk::ir::Constant>(type->ctx().int16(false), false, d));
  }

  // make sure we have only 2 operands
  CHECK(args.size() == 2);

  if (!args[0]->type()->isNumber()) {
    throw std::runtime_error("Only numeric 1st operands are supported");
  }

  // the 2nd operand does not need to be a constant
  // it can happily reference another integer column
  if (!args[1]->type()->isInteger()) {
    throw std::runtime_error("Only integer 2nd operands are supported");
  }

  // Calcite may upcast decimals in a way that is
  // incompatible with the extension function input. Play it safe and stick with the
  // argument type instead.
  auto ret_type = args[0]->type()->isDecimal() ? args[0]->type() : type;

  return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
      ret_type, fn_name, args);
}

hdk::ir::ExprPtr parseArrayFunction(const hdk::ir::ExprPtrVector& operands,
                                    const hdk::ir::Type* type) {
  if (type->isArray()) {
    // FIX-ME:  Deal with NULL arrays
    if (operands.size() > 0) {
      const auto first_element_logical_type =
          operands[0]->type()->canonicalize()->withNullable(true);

      auto diff_elem_itr =
          std::find_if(operands.begin(),
                       operands.end(),
                       [first_element_logical_type](const auto expr) {
                         return !first_element_logical_type->equal(
                             expr->type()->canonicalize()->withNullable(true));
                       });
      if (diff_elem_itr != operands.end()) {
        throw std::runtime_error(
            "Element " + std::to_string(diff_elem_itr - operands.begin()) +
            " is not of the same type as other elements of the array. Consider casting "
            "to force this condition.\nElement Type: " +
            (*diff_elem_itr)->type()->canonicalize()->withNullable(true)->toString() +
            "\nArray type: " + first_element_logical_type->toString());
      }

      auto res_type =
          type->as<hdk::ir::ArrayBaseType>()->withElemType(first_element_logical_type);
      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(res_type, operands);
    } else {
      // defaulting to valid sub-type for convenience
      auto res_type =
          type->as<hdk::ir::ArrayBaseType>()->withElemType(type->ctx().boolean());
      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(res_type, operands);
    }
  } else {
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(type, operands);
  }
}

hdk::ir::ExprPtr parseFunctionOperator(const std::string& fn_name,
                                       const hdk::ir::ExprPtrVector operands,
                                       const hdk::ir::Type* type,
                                       RelAlgDagBuilder& root_dag_builder) {
  if (fn_name == "PG_ANY"sv || fn_name == "PG_ALL"sv) {
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(type, fn_name, operands);
  }
  if (fn_name == "IS NOT NULL") {
    CHECK_EQ(operands.size(), (size_t)1);
    return makeUOper(hdk::ir::OpType::kNot,
                     makeUOper(hdk::ir::OpType::kIsNull, operands[0], type),
                     type);
  }
  if (fn_name == "LIKE"sv || fn_name == "PG_ILIKE"sv) {
    return parseLike(fn_name, operands);
  }
  if (fn_name == "REGEXP_LIKE"sv) {
    return parseRegexp(operands);
  }
  if (fn_name == "LIKELY"sv) {
    return parseLikely(operands);
  }
  if (fn_name == "UNLIKELY"sv) {
    return parseUnlikely(operands);
  }
  if (fn_name == "PG_EXTRACT"sv || fn_name == "PG_DATE_TRUNC"sv) {
    return parseExtract(fn_name, operands);
  }
  if (fn_name == "DATEADD"sv) {
    return parseDateadd(operands);
  }
  if (fn_name == "DATEDIFF"sv) {
    return parseDatediff(operands);
  }
  if (fn_name == "DATEPART"sv) {
    return parseDatepart(operands);
  }
  if (fn_name == "LENGTH"sv || fn_name == "CHAR_LENGTH"sv) {
    return parseLength(fn_name, operands);
  }
  if (fn_name == "KEY_FOR_STRING"sv) {
    return parseKeyForString(operands);
  }
  if (fn_name == "WIDTH_BUCKET"sv) {
    return parseWidthBucket(operands);
  }
  if (fn_name == "SAMPLE_RATIO"sv) {
    return parseSampleRatio(operands);
  }
  if (fn_name == "CURRENT_USER"sv) {
    return parseCurrentUser();
  }
  if (root_dag_builder.config().exec.enable_experimental_string_functions &&
      fn_name == "LOWER"sv) {
    return parseLower(operands);
  }
  if (fn_name == "CARDINALITY"sv || fn_name == "ARRAY_LENGTH"sv) {
    return parseCardinality(fn_name, operands, type);
  }
  if (fn_name == "ITEM"sv) {
    return parseItem(operands);
  }
  if (fn_name == "CURRENT_DATE"sv) {
    return parseCurrentDate(root_dag_builder.now());
  }
  if (fn_name == "CURRENT_TIME"sv) {
    return parseCurrentTime(root_dag_builder.now());
  }
  if (fn_name == "CURRENT_TIMESTAMP"sv) {
    return parseCurrentTimestamp(root_dag_builder.now());
  }
  if (fn_name == "NOW"sv) {
    return parseCurrentTimestamp(root_dag_builder.now());
  }
  if (fn_name == "DATETIME"sv) {
    return parseDatetime(operands, root_dag_builder.now());
  }
  if (fn_name == "usTIMESTAMP"sv || fn_name == "nsTIMESTAMP"sv) {
    return parseHPTLiteral(operands, type);
  }
  if (fn_name == "ABS"sv) {
    return parseAbs(operands);
  }
  if (fn_name == "SIGN"sv) {
    return parseSign(operands);
  }
  if (fn_name == "CEIL"sv || fn_name == "FLOOR"sv) {
    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        type, fn_name, operands);
  }
  if (fn_name == "ROUND"sv) {
    return parseRound(fn_name, operands, type);
  }
  if (fn_name == "DATETIME_PLUS"sv) {
    auto dt_plus = hdk::ir::makeExpr<hdk::ir::FunctionOper>(type, fn_name, operands);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
    return parseDateadd(operands);
  }
  if (fn_name == "/INT"sv) {
    CHECK_EQ(operands.size(), size_t(2));
    return Analyzer::normalizeOperExpr(
        hdk::ir::OpType::kDiv, hdk::ir::Qualifier::kOne, operands[0], operands[1]);
  }
  if (fn_name == "Reinterpret"sv) {
    CHECK_EQ(operands.size(), size_t(1));
    return operands[0];
  }
  if (fn_name == "OFFSET_IN_FRAGMENT"sv) {
    CHECK_EQ(operands.size(), size_t(0));
    return hdk::ir::makeExpr<hdk::ir::OffsetInFragment>();
  }
  if (fn_name == "ARRAY"sv) {
    // Var args; currently no check.  Possible fix-me -- can array have 0 elements?
    return parseArrayFunction(operands, type);
  }

  if (fn_name == "||"sv || fn_name == "SUBSTRING"sv) {
    auto ret_type = type->ctx().text();
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(ret_type, fn_name, operands);
  }
  // Reset possibly wrong return type of rex_function to the return
  // type of the optimal valid implementation. The return type can be
  // wrong in the case of multiple implementations of UDF functions
  // that have different return types but Calcite specifies the return
  // type according to the first implementation.
  auto args = operands;
  const hdk::ir::Type* ret_type;
  try {
    auto ext_func_sig = bind_function(fn_name, args);

    auto ext_func_args = ext_func_sig.getArgs();
    CHECK_EQ(args.size(), ext_func_args.size());
    for (size_t i = 0; i < args.size(); i++) {
      // fold casts on constants
      if (auto constant = args[i]->as<hdk::ir::Constant>()) {
        auto ext_func_arg_type = ext_arg_type_to_type(type->ctx(), ext_func_args[i]);
        if (!ext_func_arg_type->equal(args[i]->type())) {
          args[i] = constant->cast(ext_func_arg_type);
        }
      }
    }

    ret_type = ext_arg_type_to_type(type->ctx(), ext_func_sig.getRet());
  } catch (ExtensionFunctionBindingError& e) {
    LOG(WARNING) << "RelAlgTranslator::translateFunction: " << e.what();
    throw;
  }

  // By default, the extension function type will not allow nulls. If one of the arguments
  // is nullable, the extension function must also explicitly allow nulls.
  bool nullable = false;
  for (const auto& arg_expr : args) {
    if (arg_expr->type()->nullable()) {
      nullable = true;
      break;
    }
  }
  ret_type = ret_type->withNullable(nullable);

  return hdk::ir::makeExpr<hdk::ir::FunctionOper>(ret_type, fn_name, std::move(args));
}

bool isAggSupportedForType(hdk::ir::AggType agg_kind, const hdk::ir::Type* arg_type) {
  if ((agg_kind == hdk::ir::AggType::kMin || agg_kind == hdk::ir::AggType::kMax ||
       agg_kind == hdk::ir::AggType::kSum || agg_kind == hdk::ir::AggType::kAvg) &&
      !(arg_type->isNumber() || arg_type->isBoolean() || arg_type->isDateTime())) {
    return false;
  }
  if (agg_kind == hdk::ir::AggType::kQuantile) {
    return arg_type->isNumber() || arg_type->isDateTime();
  }

  return true;
}

hdk::ir::ExprPtr parseAggregateExpr(const rapidjson::Value& json_expr,
                                    RelAlgDagBuilder& root_dag_builder,
                                    const hdk::ir::ExprPtrVector& sources) {
  auto& ctx = hdk::ir::Context::defaultCtx();
  auto agg_str = json_str(field(json_expr, "agg"));
  if (agg_str == "APPROX_QUANTILE") {
    LOG(INFO) << "APPROX_QUANTILE is deprecated. Please use APPROX_PERCENTILE instead.";
  }
  auto agg_kind = to_agg_kind(agg_str);
  auto is_distinct = json_bool(field(json_expr, "distinct"));
  auto operands = indices_from_json_array(field(json_expr, "operands"));
  if (operands.size() > 1 &&
      (operands.size() != 2 || (agg_kind != hdk::ir::AggType::kApproxCountDistinct &&
                                agg_kind != hdk::ir::AggType::kApproxQuantile &&
                                agg_kind != hdk::ir::AggType::kQuantile &&
                                agg_kind != hdk::ir::AggType::kTopK))) {
    throw hdk::ir::QueryNotSupported(
        "Multiple arguments for aggregates aren't supported");
  }

  hdk::ir::ExprPtr arg_expr;
  std::shared_ptr<const hdk::ir::Constant> arg1;  // 2nd aggregate parameter
  hdk::ir::Interpolation interpolation = hdk::ir::Interpolation::kLinear;
  if (operands.size() > 0) {
    const auto operand = operands[0];
    CHECK_LT(operand, sources.size());
    CHECK_LE(operands.size(), 2u);
    arg_expr = sources[operand];

    if (agg_kind == hdk::ir::AggType::kApproxCountDistinct && operands.size() == 2) {
      arg1 = std::dynamic_pointer_cast<const hdk::ir::Constant>(sources[operands[1]]);
      if (!arg1 || !arg1->type()->isInt32() || arg1->value().intval < 1 ||
          arg1->value().intval > 100) {
        throw std::runtime_error(
            "APPROX_COUNT_DISTINCT's second parameter should be SMALLINT literal between "
            "1 and 100");
      }
    } else if (agg_kind == hdk::ir::AggType::kApproxQuantile) {
      // If second parameter is not given then APPROX_MEDIAN is assumed.
      if (operands.size() == 2) {
        arg1 = std::dynamic_pointer_cast<const hdk::ir::Constant>(
            sources[operands[1]]->cast(ctx.fp64()));
      } else {
        Datum median;
        median.doubleval = 0.5;
        arg1 = std::make_shared<hdk::ir::Constant>(ctx.fp64(), false, median);
      }
    } else if (agg_kind == hdk::ir::AggType::kTopK) {
      if (operands.size() < 2) {
        throw std::runtime_error("Missing parameter for TOP_K aggregate.");
      }
      arg1 = std::dynamic_pointer_cast<const hdk::ir::Constant>(sources[operands[1]]);
      if (!arg1 || !arg1->type()->isInteger() || arg1->value().intval == 0) {
        throw std::runtime_error(
            "TOP_K's second parameter should be non-zero integer literal");
      }
    } else if (agg_kind == hdk::ir::AggType::kQuantile) {
      if (operands.size() < 2) {
        throw std::runtime_error("Missing parameter for QUANTILE aggregate.");
      }
      arg1 = std::dynamic_pointer_cast<const hdk::ir::Constant>(sources[operands[1]]);
      if (!arg1 || !arg1->type()->isFloatingPoint() || arg1->fpVal() < 0.0 ||
          arg1->fpVal() > 1.0) {
        throw std::runtime_error(
            "QUANTILE's second parameter should be fp literal in [0, 1] range");
      }
      if (json_expr.HasMember("interpolation")) {
        auto interpolation_str = json_str(field(json_expr, "interpolation"));
        interpolation = to_interpolation(interpolation_str);
      }
    }
    auto arg_type = arg_expr->type();
    if (!isAggSupportedForType(agg_kind, arg_type)) {
      throw std::runtime_error("Aggregate on " + arg_type->toString() +
                               " is not supported yet.");
    }
  }
  auto agg_type = get_agg_type(agg_kind,
                               arg_expr.get(),
                               root_dag_builder.config().exec.group_by.bigint_count,
                               interpolation);
  return hdk::ir::makeExpr<hdk::ir::AggExpr>(
      agg_type, agg_kind, arg_expr, is_distinct, arg1, interpolation);
}

hdk::ir::ExprPtr parse_operator_expr(const rapidjson::Value& json_expr,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder,
                                     const hdk::ir::ExprPtrVector& ra_output) {
  const auto op_name = json_str(field(json_expr, "op"));
  const bool is_quantifier =
      op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? hdk::ir::OpType::kFunction : to_sql_op(op_name);
  const auto& operators_json_arr = field(json_expr, "operands");
  CHECK(operators_json_arr.IsArray());
  auto operands = parseExprArray(
      operators_json_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto type_it = json_expr.FindMember("type");
  CHECK(type_it != json_expr.MemberEnd());
  auto type = parseType(type_it->value);

  if (op == hdk::ir::OpType::kIn && json_expr.HasMember("subquery")) {
    CHECK_EQ(operands.size(), (size_t)1);
    auto subquery =
        parse_subquery_expr(json_expr, db_id, schema_provider, root_dag_builder);
    return hdk::ir::makeExpr<hdk::ir::InSubquery>(
        type->ctx().boolean(),
        operands[0],
        dynamic_cast<const hdk::ir::ScalarSubquery*>(subquery.get())->nodeShared());
  } else if (json_expr.FindMember("partition_keys") != json_expr.MemberEnd()) {
    return parseWindowFunction(json_expr,
                               op_name,
                               operands,
                               type,
                               db_id,
                               schema_provider,
                               root_dag_builder,
                               ra_output);

  } else if (op == hdk::ir::OpType::kFunction) {
    return parseFunctionOperator(op_name, operands, type, root_dag_builder);
  } else {
    CHECK_GE(operands.size(), (size_t)1);

    if (operands.size() == 1) {
      return makeUOper(op, operands[0], type);
    }

    if (auto res = maybeMakeDateExpr(op, operands, type)) {
      return res;
    }

    auto res = operands[0];
    for (size_t i = 1; i < operands.size(); ++i) {
      auto [rhs, qual] = getQuantifiedBinOperRhs(operands[i]);
      res = Analyzer::normalizeOperExpr(op, qual, res, rhs, nullptr);
    }
    return res;
  }
}

hdk::ir::ExprPtr parse_expr(const rapidjson::Value& expr,
                            int db_id,
                            SchemaProviderPtr schema_provider,
                            RelAlgDagBuilder& root_dag_builder,
                            const hdk::ir::ExprPtrVector& ra_output) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return parseInput(expr, ra_output);
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return parseLiteral(expr);
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    hdk::ir::ExprPtr res;
    const auto op_str = json_str(field(expr, "op"));
    if (op_str == std::string("CASE")) {
      res = parse_case_expr(expr, db_id, schema_provider, root_dag_builder, ra_output);
    } else if (op_str == std::string("$SCALAR_QUERY")) {
      res = parse_subquery_expr(expr, db_id, schema_provider, root_dag_builder);
    } else {
      res =
          parse_operator_expr(expr, db_id, schema_provider, root_dag_builder, ra_output);
    }
    CHECK(res);

    return res;
  }
  throw hdk::ir::QueryNotSupported("Expression node " + json_node_to_string(expr) +
                                   " not supported");
}

JoinType to_join_type(const std::string& join_type_name) {
  if (join_type_name == "inner") {
    return JoinType::INNER;
  }
  if (join_type_name == "left") {
    return JoinType::LEFT;
  }
  if (join_type_name == "semi") {
    return JoinType::SEMI;
  }
  if (join_type_name == "anti") {
    return JoinType::ANTI;
  }
  throw hdk::ir::QueryNotSupported("Join type (" + join_type_name + ") not supported");
}

void mark_nops(const std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) noexcept {
  for (auto node : nodes) {
    const auto agg_node = std::dynamic_pointer_cast<hdk::ir::Aggregate>(node);
    if (!agg_node || agg_node->getAggsCount()) {
      continue;
    }
    CHECK_EQ(size_t(1), node->inputCount());
    const auto agg_input_node =
        dynamic_cast<const hdk::ir::Aggregate*>(node->getInput(0));
    if (agg_input_node && !agg_input_node->getAggsCount() &&
        agg_node->getGroupByCount() == agg_input_node->getGroupByCount()) {
      agg_node->markAsNop();
    }
  }
}

/**
 * The InputReplacementVisitor visitor visits each node in a given relational algebra
 * expression and replaces the inputs to that expression with inputs from a different
 * node in the RA tree. Used for coalescing nodes with complex expressions.
 */
class InputReplacementVisitor : public hdk::ir::ExprRewriter {
 public:
  InputReplacementVisitor(const hdk::ir::Node* node_to_keep,
                          const hdk::ir::ExprPtrVector& exprs,
                          const hdk::ir::ExprPtrVector* groupby_exprs = nullptr)
      : node_to_keep_(node_to_keep), exprs_(exprs), groupby_exprs_(groupby_exprs) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (col_ref->node() == node_to_keep_) {
      const auto index = col_ref->index();
      CHECK_LT(index, exprs_.size());
      return visit(exprs_[index].get());
    }
    return ExprRewriter::visitColumnRef(col_ref);
  }

 private:
  const hdk::ir::Node* node_to_keep_;
  const hdk::ir::ExprPtrVector& exprs_;
  const hdk::ir::ExprPtrVector* groupby_exprs_;
};

class RANodeIterator
    : public std::vector<std::shared_ptr<hdk::ir::Node>>::const_iterator {
  using ElementType = std::shared_ptr<hdk::ir::Node>;
  using Super = std::vector<ElementType>::const_iterator;
  using Container = std::vector<ElementType>;

 public:
  enum class AdvancingMode { DUChain, InOrder };

  explicit RANodeIterator(const Container& nodes)
      : Super(nodes.begin()), owner_(nodes), nodeCount_([&nodes]() -> size_t {
        size_t non_zero_count = 0;
        for (const auto& node : nodes) {
          if (node) {
            ++non_zero_count;
          }
        }
        return non_zero_count;
      }()) {}

  explicit operator size_t() {
    return std::distance(owner_.begin(), *static_cast<Super*>(this));
  }

  RANodeIterator operator++() = delete;

  void advance(AdvancingMode mode) {
    Super& super = *this;
    switch (mode) {
      case AdvancingMode::DUChain: {
        size_t use_count = 0;
        Super only_use = owner_.end();
        for (Super nodeIt = std::next(super); nodeIt != owner_.end(); ++nodeIt) {
          if (!*nodeIt) {
            continue;
          }
          for (size_t i = 0; i < (*nodeIt)->inputCount(); ++i) {
            if ((*super) == (*nodeIt)->getAndOwnInput(i)) {
              ++use_count;
              if (1 == use_count) {
                only_use = nodeIt;
              } else {
                super = owner_.end();
                return;
              }
            }
          }
        }
        super = only_use;
        break;
      }
      case AdvancingMode::InOrder:
        for (size_t i = 0; i != owner_.size(); ++i) {
          if (!visited_.count(i)) {
            super = owner_.begin();
            std::advance(super, i);
            return;
          }
        }
        super = owner_.end();
        break;
      default:
        CHECK(false);
    }
  }

  bool allVisited() { return visited_.size() == nodeCount_; }

  const ElementType& operator*() {
    visited_.insert(size_t(*this));
    Super& super = *this;
    return *super;
  }

  const ElementType* operator->() { return &(operator*()); }

 private:
  const Container& owner_;
  const size_t nodeCount_;
  std::unordered_set<size_t> visited_;
};

bool input_can_be_coalesced(const hdk::ir::Node* parent_node,
                            const size_t index,
                            const bool first_rex_is_input) {
  if (auto agg_node = dynamic_cast<const hdk::ir::Aggregate*>(parent_node)) {
    if (index == 0 && agg_node->getGroupByCount() > 0) {
      return true;
    } else {
      // Is an aggregated target, only allow the project to be elided if the aggregate
      // target is simply passed through (i.e. if the top level expression attached to
      // the project node is a RexInput expression)
      return first_rex_is_input;
    }
  }
  return first_rex_is_input;
}

/**
 * CoalesceSecondaryProjectVisitor visits each relational algebra expression node in a
 * given input and determines whether or not the input is a candidate for coalescing
 * into the parent RA node. Intended for use only on the inputs of a Project node.
 */
class CoalesceSecondaryProjectVisitor : public ScalarExprVisitor<bool> {
 public:
  bool visitColumnRef(const hdk::ir::ColumnRef* col_ref) const final {
    // The top level expression node is checked before we apply the visitor. If we get
    // here, this input rex is a child of another rex node, and we handle the can be
    // coalesced check slightly differently
    return input_can_be_coalesced(col_ref->node(), col_ref->index(), false);
  }

  bool visitConstant(const hdk::ir::Constant*) const final { return false; }

  bool visitInSubquery(const hdk::ir::InSubquery*) const final { return false; }

  bool visitScalarSubquery(const hdk::ir::ScalarSubquery*) const final { return false; }

 protected:
  bool aggregateResult(const bool& aggregate, const bool& next_result) const final {
    return aggregate && next_result;
  }

  bool defaultResult() const final { return true; }
};

class ReplacementExprVisitor : public hdk::ir::ExprRewriter {
 public:
  ReplacementExprVisitor() {}

  ReplacementExprVisitor(
      std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements)
      : replacements_(std::move(replacements)) {}

  void addReplacement(const hdk::ir::Expr* from, hdk::ir::ExprPtr to) {
    replacements_[from] = to;
  }

  hdk::ir::ExprPtr visit(const hdk::ir::Expr* expr) override {
    auto it = replacements_.find(expr);
    if (it != replacements_.end()) {
      return it->second;
    }
    return ExprRewriter::visit(expr);
  }

 private:
  std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements_;
};

/**
 * Propagate an input backwards in the RA tree. With the exception of joins, all inputs
 * must be carried through the RA tree. This visitor takes as a parameter a source
 * projection RA Node, then checks to see if any inputs do not reference the source RA
 * node (which implies the inputs reference a node farther back in the tree). The input
 * is then backported to the source projection node, and a new input is generated which
 * references the input on the source RA node, thereby carrying the input through the
 * intermediate query step.
 */
class InputBackpropagationVisitor : public hdk::ir::ExprRewriter {
 public:
  InputBackpropagationVisitor(hdk::ir::Project* node) : node_(node) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    if (col_ref->node() != node_) {
      auto cur_index = col_ref->index();
      auto cur_source_node = col_ref->node();
      auto it = replacements_.find(std::make_pair(cur_source_node, cur_index));
      if (it != replacements_.end()) {
        return it->second;
      } else {
        std::string field_name = "";
        if (auto cur_project_node =
                dynamic_cast<const hdk::ir::Project*>(cur_source_node)) {
          field_name = cur_project_node->getFieldName(cur_index);
        }
        node_->appendInput(field_name, col_ref->shared());
        auto expr = hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            getColumnType(node_, node_->size() - 1), node_, node_->size() - 1);
        replacements_[std::make_pair(cur_source_node, cur_index)] = expr;
        return expr;
      }
    } else {
      return ExprRewriter::visitColumnRef(col_ref);
    }
  }

 protected:
  using InputReplacements =
      std::unordered_map<std::pair<const hdk::ir::Node*, unsigned>,
                         hdk::ir::ExprPtr,
                         boost::hash<std::pair<const hdk::ir::Node*, unsigned>>>;

  mutable hdk::ir::Project* node_;
  mutable InputReplacements replacements_;
};

/**
 * Detect the presence of window function operators nested inside expressions. Separate
 * the window function operator from the expression, computing the expression as a
 * subsequent step and replacing the window function operator with a RexInput. Also move
 * all input nodes to the newly created project node.
 * In pseudocode:
 * for each rex in project list:
 *    detect window function expression
 *    if window function expression:
 *        copy window function expression
 *        replace window function expression in base expression w/ input
 *        add base expression to new project node after the current node
 *        replace base expression in current project node with the window function
 expression copy
 */
void separate_window_function_expressions(
    std::vector<std::shared_ptr<hdk::ir::Node>>& nodes) {
  std::list<std::shared_ptr<hdk::ir::Node>> node_list(nodes.begin(), nodes.end());

  for (auto node_itr = node_list.begin(); node_itr != node_list.end(); ++node_itr) {
    const auto node = *node_itr;
    auto window_func_project_node = std::dynamic_pointer_cast<hdk::ir::Project>(node);
    if (!window_func_project_node) {
      continue;
    }

    // map scalar expression index in the project node to window function ptr
    std::unordered_map<size_t, hdk::ir::ExprPtr> embedded_window_function_exprs;

    // Iterate the target exprs of the project node and check for window function
    // expressions. If an embedded expression exists, save it in the
    // embedded_window_function_expressions map and split the expression into a window
    // function expression and a parent expression in a subsequent project node
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      auto expr = window_func_project_node->getExpr(i);
      if (hdk::ir::isWindowFunctionExpr(expr.get())) {
        // top level window function exprs are fine
        continue;
      }

      std::list<const hdk::ir::Expr*> window_function_exprs =
          ExprByPredicateCollector::collect(expr.get(), hdk::ir::isWindowFunctionExpr);
      if (!window_function_exprs.empty()) {
        const auto ret = embedded_window_function_exprs.insert(
            std::make_pair(i, window_function_exprs.front()->shared_from_this()));
        CHECK(ret.second);
      }
    }

    if (!embedded_window_function_exprs.empty()) {
      hdk::ir::ExprPtrVector new_exprs;

      auto window_func_exprs = window_func_project_node->getExprs();
      for (size_t rex_idx = 0; rex_idx < window_func_exprs.size(); ++rex_idx) {
        const auto embedded_window_func_expr_pair =
            embedded_window_function_exprs.find(rex_idx);
        if (embedded_window_func_expr_pair == embedded_window_function_exprs.end()) {
          new_exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
              window_func_project_node->getExpr(rex_idx)->type(),
              window_func_project_node.get(),
              rex_idx));
        } else {
          const auto window_func_expr_idx = embedded_window_func_expr_pair->first;
          CHECK_LT(window_func_expr_idx, window_func_exprs.size());

          const auto& window_func_expr = embedded_window_func_expr_pair->second;
          auto window_func_parent_expr = window_func_exprs[window_func_expr_idx].get();

          // Replace window func expr with ColumnRef
          auto window_func_result_input_expr =
              hdk::ir::makeExpr<hdk::ir::ColumnRef>(window_func_expr->type(),
                                                    window_func_project_node.get(),
                                                    window_func_expr_idx);
          std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements;
          replacements[window_func_expr.get()] = window_func_result_input_expr;
          ReplacementExprVisitor visitor(std::move(replacements));
          auto new_parent_expr = visitor.visit(window_func_parent_expr);

          // Put the parent expr in the new scalar exprs
          new_exprs.emplace_back(std::move(new_parent_expr));

          // Put the window func expr in cur scalar exprs
          window_func_exprs[window_func_expr_idx] = window_func_expr;
        }
      }

      CHECK_EQ(window_func_exprs.size(), new_exprs.size());
      window_func_project_node->setExpressions(std::move(window_func_exprs));

      // Ensure any inputs from the node containing the expression (the "new" node)
      // exist on the window function project node, e.g. if we had a binary operation
      // involving an aggregate value or column not included in the top level
      // projection list.
      InputBackpropagationVisitor visitor(window_func_project_node.get());
      for (size_t i = 0; i < new_exprs.size(); i++) {
        if (dynamic_cast<const hdk::ir::ColumnRef*>(new_exprs[i].get())) {
          // ignore top level inputs, these were copied directly from the previous
          // node
          continue;
        }
        new_exprs[i] = visitor.visit(new_exprs[i].get());
      }

      // Build the new project node and insert it into the list after the project node
      // containing the window function
      auto new_project =
          std::make_shared<hdk::ir::Project>(std::move(new_exprs),
                                             window_func_project_node->getFields(),
                                             window_func_project_node);
      node_list.insert(std::next(node_itr), new_project);

      // Rebind all the following inputs
      for (auto rebind_itr = std::next(node_itr, 2); rebind_itr != node_list.end();
           rebind_itr++) {
        (*rebind_itr)->replaceInput(window_func_project_node, new_project);
      }
    }
  }
  nodes.assign(node_list.begin(), node_list.end());
}

int64_t get_int_literal_field(const rapidjson::Value& obj,
                              const char field[],
                              const int64_t default_val) noexcept {
  const auto it = obj.FindMember(field);
  if (it == obj.MemberEnd()) {
    return default_val;
  }
  auto expr = parseLiteral(it->value);
  CHECK(expr->type()->isInteger());
  return dynamic_cast<const hdk::ir::Constant*>(expr.get())->intVal();
}

void check_empty_inputs_field(const rapidjson::Value& node) noexcept {
  const auto& inputs_json = field(node, "inputs");
  CHECK(inputs_json.IsArray() && !inputs_json.Size());
}

TableInfoPtr getTableFromScanNode(int db_id,
                                  SchemaProviderPtr schema_provider,
                                  const rapidjson::Value& scan_ra) {
  const auto& table_json = field(scan_ra, "table");
  CHECK(table_json.IsArray());
  CHECK_EQ(unsigned(2), table_json.Size());
  auto info = schema_provider->getTableInfo(db_id, table_json[1].GetString());
  // If table wasn't found in the default database, then try search in the
  // result set registry.
  if (!info) {
    info = schema_provider->getTableInfo(hdk::ResultSetRegistry::DB_ID,
                                         table_json[1].GetString());
  }
  CHECK(info);
  return info;
}

std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) {
  const auto& fields_json = field(scan_ra, "fieldNames");
  return strings_from_json_array(fields_json);
}

hdk::ir::ExprPtrVector genColumnRefs(const hdk::ir::Node* node, size_t count) {
  hdk::ir::ExprPtrVector res;
  res.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    res.emplace_back(
        hdk::ir::makeExpr<hdk::ir::ColumnRef>(getColumnType(node, i), node, i));
  }
  return res;
}

// Recognize the left-deep join tree pattern with an optional filter as root
// with `node` as the parent of the join sub-tree. On match, return the root
// of the recognized tree (either the filter node or the outermost join).
std::shared_ptr<const hdk::ir::Node> get_join_root(
    const std::shared_ptr<hdk::ir::Node>& node) {
  const auto join_filter = dynamic_cast<const hdk::ir::Filter*>(node.get());
  if (join_filter) {
    const auto join = dynamic_cast<const hdk::ir::Join*>(join_filter->getInput(0));
    if (!join) {
      return nullptr;
    }
    if (join->getJoinType() == JoinType::INNER || join->getJoinType() == JoinType::SEMI ||
        join->getJoinType() == JoinType::ANTI) {
      return node;
    }
  }
  if (!node || node->inputCount() != 1) {
    return nullptr;
  }
  const auto join = dynamic_cast<const hdk::ir::Join*>(node->getInput(0));
  if (!join) {
    return nullptr;
  }
  return node->getAndOwnInput(0);
}

}  // namespace

/**
 * Join can become an execution point in a query. This is inconvenient because
 * Join node doesn't allow to filter-out unused columns. Here we insert projections
 * that would help to avoid Joins as execution points and enable dead columns
 * elimination.
 */
void insert_join_projections(std::vector<hdk::ir::NodePtr>& nodes) {
  std::list<hdk::ir::NodePtr> node_list(nodes.begin(), nodes.end());
  auto du_web = build_du_web(nodes);
  for (auto node_it = node_list.begin(); node_it != node_list.end(); ++node_it) {
    auto& node = *node_it;
    auto join = node->as<hdk::ir::Join>();
    if (!join) {
      continue;
    }

    CHECK(du_web.count(join));
    auto join_users = du_web.at(join);
    if (join_users.empty()) {
      continue;
    }

    bool insert_projection = false;
    if (join_users.size() > 1) {
      insert_projection = true;
    } else {
      auto join_user = *join_users.begin();
      if (join_user->is<hdk::ir::Join>() && join == join_user->getInput(0)) {
        insert_projection = false;
      } else {
        insert_projection = !join_user->is<hdk::ir::Project>();
      }
    }

    // Here we insert a projection for all join columns. We assume that dead column
    // elimination will be used to remove unused ones.
    if (insert_projection) {
      hdk::ir::ExprPtrVector exprs;
      std::vector<std::string> fields;
      exprs.reserve(join->size());
      fields.reserve(join->size());
      for (unsigned i = 0; i < (unsigned)join->size(); ++i) {
        exprs.emplace_back(getNodeColumnRef(join, i));
        fields.emplace_back("expr_" + std::to_string(i));
      }
      auto new_project =
          std::make_shared<hdk::ir::Project>(std::move(exprs), std::move(fields), node);
      auto next_it = node_it;
      ++next_it;
      node_list.insert(next_it, new_project);

      for (auto join_user : join_users) {
        const_cast<hdk::ir::Node*>(join_user)->replaceInput(node, new_project);
      }
    }
  }

  if (node_list.size() != nodes.size()) {
    nodes.assign(node_list.begin(), node_list.end());
  }
}

namespace details {

class RelAlgDispatcher {
 public:
  RelAlgDispatcher(int db_id, SchemaProviderPtr schema_provider)
      : db_id_(db_id), schema_provider_(schema_provider) {}

  std::vector<std::shared_ptr<hdk::ir::Node>> run(const rapidjson::Value& rels,
                                                  RelAlgDagBuilder& root_dag_builder) {
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      std::shared_ptr<hdk::ir::Node> ra_node = nullptr;
      const auto rel_op = json_str(field(crt_node, "relOp"));
      if (rel_op == std::string("EnumerableTableScan") ||
          rel_op == std::string("LogicalTableScan")) {
        ra_node = dispatchTableScan(crt_node);
      } else if (rel_op == std::string("LogicalProject")) {
        ra_node = dispatchProject(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalFilter")) {
        ra_node = dispatchFilter(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalAggregate")) {
        ra_node = dispatchAggregate(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalJoin")) {
        ra_node = dispatchJoin(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalSort")) {
        ra_node = dispatchSort(crt_node);
      } else if (rel_op == std::string("LogicalValues")) {
        ra_node = dispatchLogicalValues(crt_node);
      } else if (rel_op == std::string("LogicalUnion")) {
        ra_node = dispatchUnion(crt_node);
      } else {
        throw hdk::ir::QueryNotSupported(std::string("Node ") + rel_op +
                                         " not supported yet");
      }
      nodes_.push_back(ra_node);
    }

    return std::move(nodes_);
  }

 private:
  std::shared_ptr<hdk::ir::Scan> dispatchTableScan(const rapidjson::Value& scan_ra) {
    check_empty_inputs_field(scan_ra);
    CHECK(scan_ra.IsObject());
    const auto tinfo = getTableFromScanNode(db_id_, schema_provider_, scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    std::vector<ColumnInfoPtr> infos;
    infos.reserve(field_names.size());
    for (auto& col_name : field_names) {
      infos.emplace_back(schema_provider_->getColumnInfo(*tinfo, col_name));
      CHECK(infos.back());
    }
    return std::make_shared<hdk::ir::Scan>(tinfo, std::move(infos));
  }

  std::shared_ptr<hdk::ir::Project> dispatchProject(const rapidjson::Value& proj_ra,
                                                    RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(proj_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto& exprs_json = field(proj_ra, "exprs");
    CHECK(exprs_json.IsArray());
    hdk::ir::ExprPtrVector exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End();
         ++exprs_json_it) {
      exprs.emplace_back(parse_expr(*exprs_json_it,
                                    db_id_,
                                    schema_provider_,
                                    root_dag_builder,
                                    getNodeColumnRefs(inputs[0].get())));
    }
    if (!exprs.size()) {
      throw hdk::ir::QueryNotSupported("Empty projections are not allowed.");
    }
    const auto& fields = field(proj_ra, "fields");
    return std::make_shared<hdk::ir::Project>(
        std::move(exprs), strings_from_json_array(fields), inputs.front());
  }

  std::shared_ptr<hdk::ir::Filter> dispatchFilter(const rapidjson::Value& filter_ra,
                                                  RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(filter_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto id = node_id(filter_ra);
    CHECK(id);
    auto condition_expr = parse_expr(field(filter_ra, "condition"),
                                     db_id_,
                                     schema_provider_,
                                     root_dag_builder,
                                     getNodeColumnRefs(inputs[0].get()));
    return std::make_shared<hdk::ir::Filter>(std::move(condition_expr), inputs.front());
  }

  std::shared_ptr<hdk::ir::Aggregate> dispatchAggregate(
      const rapidjson::Value& agg_ra,
      RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(agg_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto fields = strings_from_json_array(field(agg_ra, "fields"));
    const auto group = indices_from_json_array(field(agg_ra, "group"));
    for (size_t i = 0; i < group.size(); ++i) {
      CHECK_EQ(i, group[i]);
    }
    if (agg_ra.HasMember("groups") || agg_ra.HasMember("indicator")) {
      throw hdk::ir::QueryNotSupported("GROUP BY extensions not supported");
    }
    const auto& aggs_json_arr = field(agg_ra, "aggs");
    CHECK(aggs_json_arr.IsArray());
    hdk::ir::ExprPtrVector input_exprs = getInputExprsForAgg(inputs[0].get());
    hdk::ir::ExprPtrVector aggs;
    for (auto aggs_json_arr_it = aggs_json_arr.Begin();
         aggs_json_arr_it != aggs_json_arr.End();
         ++aggs_json_arr_it) {
      auto agg = parseAggregateExpr(*aggs_json_arr_it, root_dag_builder, input_exprs);
      aggs.push_back(agg);
    }
    auto agg_node = std::make_shared<hdk::ir::Aggregate>(
        group.size(), std::move(aggs), fields, inputs.front());
    return agg_node;
  }

  std::shared_ptr<hdk::ir::Join> dispatchJoin(const rapidjson::Value& join_ra,
                                              RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(join_ra);
    CHECK_EQ(size_t(2), inputs.size());
    const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
    auto ra_outputs = genColumnRefs(inputs[0].get(), inputs[0]->size());
    const auto ra_outputs2 = genColumnRefs(inputs[1].get(), inputs[1]->size());
    ra_outputs.insert(ra_outputs.end(), ra_outputs2.begin(), ra_outputs2.end());
    auto condition = parse_expr(field(join_ra, "condition"),
                                db_id_,
                                schema_provider_,
                                root_dag_builder,
                                ra_outputs);
    auto join_node = std::make_shared<hdk::ir::Join>(
        inputs[0], inputs[1], std::move(condition), join_type);
    return join_node;
  }

  std::shared_ptr<hdk::ir::Sort> dispatchSort(const rapidjson::Value& sort_ra) {
    const auto inputs = getRelAlgInputs(sort_ra);
    CHECK_EQ(size_t(1), inputs.size());
    std::vector<hdk::ir::SortField> collation;
    const auto& collation_arr = field(sort_ra, "collation");
    CHECK(collation_arr.IsArray());
    for (auto collation_arr_it = collation_arr.Begin();
         collation_arr_it != collation_arr.End();
         ++collation_arr_it) {
      const size_t field_idx = json_i64(field(*collation_arr_it, "field"));
      const auto sort_dir = parse_sort_direction(*collation_arr_it);
      const auto null_pos = parse_nulls_position(*collation_arr_it);
      collation.emplace_back(field_idx, sort_dir, null_pos);
    }
    auto limit = get_int_literal_field(sort_ra, "fetch", -1);
    const auto offset = get_int_literal_field(sort_ra, "offset", 0);
    auto ret = std::make_shared<hdk::ir::Sort>(
        collation, limit > 0 ? limit : 0, offset, inputs.front());
    ret->setEmptyResult(limit == 0);
    return ret;
  }

  std::vector<hdk::ir::TargetMetaInfo> parseTupleType(
      const rapidjson::Value& tuple_type_arr) {
    CHECK(tuple_type_arr.IsArray());
    std::vector<hdk::ir::TargetMetaInfo> tuple_type;
    for (auto tuple_type_arr_it = tuple_type_arr.Begin();
         tuple_type_arr_it != tuple_type_arr.End();
         ++tuple_type_arr_it) {
      auto component_type = parseType(*tuple_type_arr_it);
      const auto component_name = json_str(field(*tuple_type_arr_it, "name"));
      tuple_type.emplace_back(component_name, component_type);
    }
    return tuple_type;
  }

  std::shared_ptr<hdk::ir::LogicalValues> dispatchLogicalValues(
      const rapidjson::Value& logical_values_ra) {
    const auto& tuple_type_arr = field(logical_values_ra, "type");
    std::vector<hdk::ir::TargetMetaInfo> tuple_type = parseTupleType(tuple_type_arr);
    const auto& inputs_arr = field(logical_values_ra, "inputs");
    CHECK(inputs_arr.IsArray());
    const auto& tuples_arr = field(logical_values_ra, "tuples");
    CHECK(tuples_arr.IsArray());

    if (inputs_arr.Size()) {
      throw hdk::ir::QueryNotSupported("Inputs not supported in logical values yet.");
    }

    std::vector<hdk::ir::ExprPtrVector> values;
    if (tuples_arr.Size()) {
      for (const auto& row : tuples_arr.GetArray()) {
        CHECK(row.IsArray());
        const auto values_json = row.GetArray();
        if (!values.empty()) {
          CHECK_EQ(values[0].size(), values_json.Size());
        }
        values.emplace_back(hdk::ir::ExprPtrVector{});
        for (const auto& value : values_json) {
          CHECK(value.IsObject());
          CHECK(value.HasMember("literal"));
          values.back().emplace_back(parseLiteral(value));
        }
      }
    }

    return std::make_shared<hdk::ir::LogicalValues>(tuple_type, std::move(values));
  }

  std::shared_ptr<hdk::ir::LogicalUnion> dispatchUnion(
      const rapidjson::Value& logical_union_ra) {
    auto inputs = getRelAlgInputs(logical_union_ra);
    auto const& all_type_bool = field(logical_union_ra, "all");
    CHECK(all_type_bool.IsBool());
    return std::make_shared<hdk::ir::LogicalUnion>(std::move(inputs),
                                                   all_type_bool.GetBool());
  }

  hdk::ir::NodeInputs getRelAlgInputs(const rapidjson::Value& node) {
    if (node.HasMember("inputs")) {
      const auto str_input_ids = strings_from_json_array(field(node, "inputs"));
      hdk::ir::NodeInputs ra_inputs;
      for (const auto& str_id : str_input_ids) {
        ra_inputs.push_back(nodes_[std::stoi(str_id)]);
      }
      return ra_inputs;
    }
    return {prev(node)};
  }

  std::pair<std::string, std::string> getKVOptionPair(std::string& str, size_t& pos) {
    auto option = str.substr(0, pos);
    std::string delim = "=";
    size_t delim_pos = option.find(delim);
    auto key = option.substr(0, delim_pos);
    auto val = option.substr(delim_pos + 1, option.length());
    str.erase(0, pos + delim.length() + 1);
    return {key, val};
  }

  hdk::ir::NodePtr prev(const rapidjson::Value& crt_node) {
    const auto id = node_id(crt_node);
    CHECK(id);
    CHECK_EQ(static_cast<size_t>(id), nodes_.size());
    return nodes_.back();
  }

  int db_id_;
  SchemaProviderPtr schema_provider_;
  std::vector<std::shared_ptr<hdk::ir::Node>> nodes_;
};

}  // namespace details

RelAlgDagBuilder::RelAlgDagBuilder(RelAlgDagBuilder& root_dag_builder,
                                   const rapidjson::Value& query_ast,
                                   int db_id,
                                   SchemaProviderPtr schema_provider)
    : hdk::ir::QueryDag(root_dag_builder.config_, root_dag_builder.now())
    , db_id_(db_id)
    , schema_provider_(schema_provider) {
  build(query_ast, root_dag_builder);
}

RelAlgDagBuilder::RelAlgDagBuilder(const rapidjson::Value& query_ast,
                                   int db_id,
                                   SchemaProviderPtr schema_provider,
                                   ConfigPtr config)
    : hdk::ir::QueryDag(config), db_id_(db_id), schema_provider_(schema_provider) {
  build(query_ast, *this);
}

RelAlgDagBuilder::RelAlgDagBuilder(const std::string& query_ra,
                                   int db_id,
                                   SchemaProviderPtr schema_provider,
                                   ConfigPtr config)
    : hdk::ir::QueryDag(config), db_id_(db_id), schema_provider_(schema_provider) {
  rapidjson::Document query_ast;
  query_ast.Parse(query_ra.c_str());
  VLOG(2) << "Parsing query RA JSON: " << query_ra;
  if (query_ast.HasParseError()) {
    query_ast.GetParseError();
    LOG(ERROR) << "Failed to parse RA tree from Calcite (offset "
               << query_ast.GetErrorOffset() << "):\n"
               << rapidjson::GetParseError_En(query_ast.GetParseError());
    VLOG(1) << "Failed to parse query RA: " << query_ra;
    throw std::runtime_error(
        "Failed to parse relational algebra tree. Possible query syntax error.");
  }
  CHECK(query_ast.IsObject());
  build(query_ast, *this);
}

void RelAlgDagBuilder::build(const rapidjson::Value& query_ast,
                             RelAlgDagBuilder& lead_dag_builder) {
  const auto& rels = field(query_ast, "rels");
  CHECK(rels.IsArray());
  try {
    nodes_ =
        details::RelAlgDispatcher(db_id_, schema_provider_).run(rels, lead_dag_builder);
  } catch (const hdk::ir::QueryNotSupported&) {
    throw;
  }
  CHECK(!nodes_.empty());

  mark_nops(nodes_);
  simplify_sort(nodes_);
  sink_projected_boolean_expr_to_join(nodes_);
  eliminate_identical_copy(nodes_);
  insert_join_projections(nodes_);
  fold_filters(nodes_);
  bool has_filtered_join = false;
  for (const auto& node : nodes_) {
    const auto join_root = get_join_root(node);
    // The filter which starts a left-deep join pattern must not be coalesced
    // since it contains (part of) the join condition.
    if (join_root) {
      if (std::dynamic_pointer_cast<const hdk::ir::Filter>(join_root)) {
        has_filtered_join = true;
        break;
      }
    }
  }
  if (has_filtered_join) {
    hoist_filter_cond_to_cross_join(nodes_);
  }
  eliminate_dead_columns(nodes_);
  eliminate_dead_subqueries(subqueries_, nodes_.back().get());
  separate_window_function_expressions(nodes_);
  CHECK(nodes_.size());
  CHECK(nodes_.back().use_count() == 1);
  root_ = nodes_.back();
}

// Return tree with depth represented by indentations.
std::string tree_string(const hdk::ir::Node* ra, const size_t depth) {
  std::string result = std::string(2 * depth, ' ') + ::toString(ra) + '\n';
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    result += tree_string(ra->getInput(i), depth + 1);
  }
  return result;
}

hdk::ir::ExprPtrVector getInputExprsForAgg(const hdk::ir::Node* node) {
  hdk::ir::ExprPtrVector res;
  res.reserve(node->size());
  auto project = dynamic_cast<const hdk::ir::Project*>(node);
  if (project) {
    const auto& exprs = project->getExprs();
    for (unsigned col_idx = 0; col_idx < static_cast<unsigned>(exprs.size()); ++col_idx) {
      auto& expr = exprs[col_idx];
      if (dynamic_cast<const hdk::ir::Constant*>(expr.get())) {
        res.emplace_back(expr);
      } else {
        res.emplace_back(
            hdk::ir::makeExpr<hdk::ir::ColumnRef>(expr->type(), node, col_idx));
      }
    }
  } else if (node->is<hdk::ir::LogicalValues>() || node->is<hdk::ir::Aggregate>() ||
             node->is<hdk::ir::LogicalUnion>()) {
    res = getNodeColumnRefs(node);
  } else {
    CHECK(false) << "Unexpected node: " << node->toString();
  }

  return res;
}
