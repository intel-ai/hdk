/*
 * Copyright 2021 OmniSci, Inc.
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
 * @file    Analyzer.h
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Defines data structures for the semantic analysis phase of query processing
 **/
#ifndef ANALYZER_H
#define ANALYZER_H

#include "IR/Expr.h"
#include "Logger/Logger.h"
#include "SchemaMgr/ColumnInfo.h"
#include "Shared/sqldefs.h"
#include "Shared/sqltypes.h"
#include "StringDictionary/StringDictionary.h"

#include <cstdint>
#include <iostream>
#include <list>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Analyzer {

const hdk::ir::Type* common_numeric_type(const hdk::ir::Type* type1,
                                         const hdk::ir::Type* type2);
const hdk::ir::Type* common_string_type(const hdk::ir::Type* type1,
                                        const hdk::ir::Type* type2);

hdk::ir::ExprPtr analyzeIntValue(const int64_t intval);

hdk::ir::ExprPtr analyzeFixedPtValue(const int64_t numericval,
                                     const int scale,
                                     const int precision);

hdk::ir::ExprPtr analyzeStringValue(const std::string& stringval);

hdk::ir::ExprPtr normalizeOperExpr(hdk::ir::OpType optype,
                                   hdk::ir::Qualifier qual,
                                   hdk::ir::ExprPtr left_expr,
                                   hdk::ir::ExprPtr right_expr,
                                   const StringDictionaryProvider* executor = nullptr);

hdk::ir::ExprPtr normalizeCaseExpr(
    const std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>>&,
    const hdk::ir::ExprPtr,
    const StringDictionaryProvider* executor = nullptr);

hdk::ir::ExprPtr getLikeExpr(hdk::ir::ExprPtr arg_expr,
                             hdk::ir::ExprPtr like_expr,
                             hdk::ir::ExprPtr escape_expr,
                             const bool is_ilike,
                             const bool is_not);

hdk::ir::ExprPtr getRegexpExpr(hdk::ir::ExprPtr arg_expr,
                               hdk::ir::ExprPtr pattern_expr,
                               hdk::ir::ExprPtr escape_expr,
                               const bool is_not);

hdk::ir::ExprPtr getUserLiteral(const std::string&);

hdk::ir::ExprPtr getTimestampLiteral(const int64_t);

}  // namespace Analyzer

inline std::shared_ptr<const hdk::ir::Var> var_ref(const hdk::ir::Expr* expr,
                                                   const hdk::ir::Var::WhichRow which_row,
                                                   const int varno) {
  if (const auto col_expr = dynamic_cast<const hdk::ir::ColumnVar*>(expr)) {
    return hdk::ir::makeExpr<hdk::ir::Var>(
        col_expr->columnInfo(), col_expr->rteIdx(), which_row, varno);
  }
  return hdk::ir::makeExpr<hdk::ir::Var>(expr->type(), which_row, varno);
}

// Remove a cast operator if present.
hdk::ir::ExprPtr remove_cast(const hdk::ir::ExprPtr& expr);

#endif  // ANALYZER_H
