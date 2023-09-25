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

#pragma once

#include "IR/ExprCollector.h"
#include "Logger/Logger.h"
#include "QueryEngine/ScalarExprVisitor.h"

class TransientStringLiteralsVisitor : public hdk::ir::ExprVisitor<void> {
 public:
  TransientStringLiteralsVisitor(StringDictionaryProxy* sdp, Executor* executor)
      : sdp_(sdp), executor_(executor) {
    CHECK(sdp);
  }

  void visitConstant(const hdk::ir::Constant* constant) override {
    if (constant->type()->isString() && !constant->isNull()) {
      CHECK(constant->value().stringval);
      sdp_->getOrAdd(*constant->value().stringval);
    }
  }

  // visitUOper is for handling casts between dictionary encoded text
  // columns that do not share string dictionaries. For these
  // we need to run the translation again on the aggregator
  // so that we know how to interpret the transient literals added
  // by the leaves via string-to-string casts

  // Todo(todd): It is inefficient to do the same translation on
  // the aggregator and each of the leaves, explore storing these
  // translations/literals on the remote dictionary server instead
  // so the translation happens once and only once

  void visitUOper(const hdk::ir::UOper* uoper) override {
    visit(uoper->operand());
    auto uoper_type = uoper->type();
    auto operand_type = uoper->operand()->type();
    if (!(uoper->isCast() && uoper_type->isExtDictionary() &&
          operand_type->isExtDictionary())) {
      // If we are not casting from a dictionary-encoded string
      // to a dictionary-encoded string
      return;
    }
    auto uoper_dict_id = uoper_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto operand_dict_id = operand_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    if (uoper_dict_id != sdp_->getBaseDictionary()->getDictId()) {
      // If we are not casting to our dictionary (sdp_
      return;
    }
    if (uoper_dict_id == operand_dict_id) {
      // If cast is inert, i.e. source and destination dict ids are same
      return;
    }
    if (uoper->isDictIntersection()) {
      // Intersection translations don't add transients to the dest proxy,
      // and hence can be ignored for the purposes of populating transients
      return;
    }
    executor_->getStringProxyTranslationMap(
        operand_dict_id,
        uoper_dict_id,
        RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
        executor_->getRowSetMemoryOwner(),
        true);  // with_generation
  }

 protected:
  StringDictionaryProxy* sdp_;
  Executor* executor_;
};

class TransientDictIdCollector
    : public hdk::ir::ExprCollector<int, TransientDictIdCollector> {
 public:
  void visit(const hdk::ir::Expr* expr) override {
    result_ = -1;
    ExprVisitor::visit(expr);
  }

 protected:
  void visitUOper(const hdk::ir::UOper* uoper) override {
    auto expr_type = uoper->type();
    if (uoper->isCast() && expr_type->isExtDictionary()) {
      result_ = expr_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    } else {
      ExprCollector::visitUOper(uoper);
    }
  }

  void visitCaseExpr(const hdk::ir::CaseExpr* case_expr) override {
    auto expr_type = case_expr->type();
    if (expr_type->isExtDictionary()) {
      result_ = expr_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    } else {
      ExprCollector::visitCaseExpr(case_expr);
    }
  }
};
