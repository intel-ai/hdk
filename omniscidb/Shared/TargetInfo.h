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

/**
 * @file    TargetInfo.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_TARGETINFO_H
#define QUERYENGINE_TARGETINFO_H

#include "sqldefs.h"
#include "sqltypes.h"

#include "IR/Expr.h"
#include "IR/TypeUtils.h"

inline const hdk::ir::AggExpr* cast_to_agg_expr(const hdk::ir::Expr* target_expr) {
  return dynamic_cast<const hdk::ir::AggExpr*>(target_expr);
}

inline const hdk::ir::AggExpr* cast_to_agg_expr(const hdk::ir::ExprPtr target_expr) {
  return dynamic_cast<const hdk::ir::AggExpr*>(target_expr.get());
}

struct TargetInfo {
  bool is_agg;
  hdk::ir::AggType agg_kind;
  const hdk::ir::Type* type;
  const hdk::ir::Type* agg_arg_type;
  bool skip_null_val;
  bool is_distinct;
  int topk_param;
  bool topk_inline_buffer;
#ifndef __CUDACC__
 public:
  inline std::string toString() const {
    auto result = std::string("TargetInfo(");
    result += "is_agg=" + std::string(is_agg ? "true" : "false") + ", ";
    result += "agg_kind=" + ::toString(agg_kind) + ", ";
    result += "type=" + type->toString() + ", ";
    result += "agg_arg_type=" +
              (agg_arg_type ? agg_arg_type->toString() : std::string("null")) + ", ";
    result += "skip_null_val=" + std::string(skip_null_val ? "true" : "false") + ", ";
    result += "is_distinct=" + std::string(is_distinct ? "true" : "false") + ", ";
    result += "topk_param=" + ::toString(topk_param) + ", ";
    result +=
        "topk_inline_buffer=" + std::string(topk_inline_buffer ? "true" : "false") + ")";
    return result;
  }
#endif
};

/**
 * Returns true if the aggregate function always returns a value in the domain of the
 * argument. Returns false otherwise.
 */
inline bool is_agg_domain_range_equivalent(hdk::ir::AggType agg_kind) {
  switch (agg_kind) {
    case hdk::ir::AggType::kMin:
    case hdk::ir::AggType::kMax:
    case hdk::ir::AggType::kSingleValue:
    case hdk::ir::AggType::kSample:
      return true;
    default:
      break;
  }
  return false;
}

template <class PointerType>
inline TargetInfo get_target_info(const PointerType target_expr,
                                  const bool bigint_count) {
  auto& ctx = target_expr->type()->ctx();
  const hdk::ir::AggExpr* agg_expr = cast_to_agg_expr(target_expr);
  bool nullable = target_expr->type()->nullable();
  if (!agg_expr) {
    auto target_type = target_expr->type()->canonicalize();
    return {false, hdk::ir::AggType::kMin, target_type, nullptr, false, false};
  }
  const auto agg_type = agg_expr->aggType();
  const auto agg_arg = agg_expr->arg();
  if (!agg_arg) {
    CHECK_EQ(hdk::ir::AggType::kCount, agg_type);
    CHECK(!agg_expr->isDistinct());
    return {true,
            hdk::ir::AggType::kCount,
            ctx.integer(bigint_count ? 8 : 4, nullable),
            nullptr,
            false,
            false,
            0,
            false};
  }

  auto agg_arg_type = agg_arg->type();
  bool is_distinct{false};
  if (agg_expr->aggType() == hdk::ir::AggType::kCount) {
    is_distinct = agg_expr->isDistinct();
  }

  if (agg_type == hdk::ir::AggType::kAvg) {
    // Upcast the target type for AVG, so that the integer argument does not overflow the
    // sum
    return {
        true,
        agg_expr->aggType(),
        agg_arg_type->isInteger() ? ctx.int64(agg_arg_type->nullable()) : agg_arg_type,
        agg_arg_type,
        agg_arg_type->nullable(),
        is_distinct,
        0,
        false};
  }

  int topk_param =
      agg_type == hdk::ir::AggType::kTopK
          ? static_cast<int>(agg_expr->arg1()->as<hdk::ir::Constant>()->intVal())
          : 0;
  bool topk_inline_buffer =
      topk_param && (agg_expr->arg()->type()->canonicalSize() * std::abs(topk_param) <=
                     sizeof(int64_t*));
  return {true,
          agg_expr->aggType(),
          agg_type == hdk::ir::AggType::kCount
              ? ctx.integer((is_distinct || bigint_count) ? 8 : 4, nullable)
              : agg_expr->type(),
          agg_arg_type,
          agg_type == hdk::ir::AggType::kCount &&
                  (agg_arg_type->isString() || agg_arg_type->isArray())
              ? false
              : agg_arg_type->nullable(),
          is_distinct,
          topk_param,
          topk_inline_buffer};
}

inline bool is_distinct_target(const TargetInfo& target_info) {
  return target_info.is_distinct ||
         target_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct;
}

inline bool takes_float_argument(const TargetInfo& target_info) {
  return target_info.is_agg &&
         (target_info.agg_kind == hdk::ir::AggType::kAvg ||
          target_info.agg_kind == hdk::ir::AggType::kSum ||
          target_info.agg_kind == hdk::ir::AggType::kMin ||
          target_info.agg_kind == hdk::ir::AggType::kMax ||
          target_info.agg_kind == hdk::ir::AggType::kSingleValue) &&
         target_info.agg_arg_type->isFp32();
}

#endif  // QUERYENGINE_TARGETINFO_H
