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

#include "CodeGenerator.h"
#include "Execute.h"

// Code generation routines and helpers for basic arithmetic and unary minus.
namespace {

std::string numeric_or_time_interval_type_name(const hdk::ir::Type* type1,
                                               const hdk::ir::Type* type2) {
  if (type2->isInterval()) {
    return numeric_type_name(type2);
  }
  return numeric_type_name(type1);
}

}  // namespace

llvm::Value* CodeGenerator::codegenArith(const hdk::ir::BinOper* bin_oper,
                                         const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = bin_oper->opType();
  CHECK(hdk::ir::isArithmetic(optype));
  const auto lhs = bin_oper->leftOperand();
  const auto rhs = bin_oper->rightOperand();
  const auto& lhs_type = lhs->type();
  const auto& rhs_type = rhs->type();

  if (lhs_type->isDecimal() && rhs_type->isDecimal() && optype == hdk::ir::OpType::kDiv) {
    const auto ret = codegenDeciDiv(bin_oper, co);
    if (ret) {
      return ret;
    }
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  // Handle operations when a time interval operand is involved, an operation
  // between an integer and a time interval isn't normalized by the analyzer.
  if (lhs_type->isInterval()) {
    rhs_lv = codegenCastBetweenIntTypes(rhs_lv, rhs_type, lhs_type);
  } else if (rhs_type->isInterval()) {
    lhs_lv = codegenCastBetweenIntTypes(lhs_lv, lhs_type, rhs_type);
  } else {
    CHECK_EQ(lhs_type->id(), rhs_type->id());
  }
  if (lhs_type->isInteger() || lhs_type->isDecimal() || lhs_type->isInterval()) {
    return codegenIntArith(bin_oper, lhs_lv, rhs_lv, co);
  }
  if (lhs_type->isFloatingPoint()) {
    return codegenFpArith(bin_oper, lhs_lv, rhs_lv);
  }
  CHECK(false);
  return nullptr;
}

// Handle integer or integer-like (decimal, time, date) operand types.
llvm::Value* CodeGenerator::codegenIntArith(const hdk::ir::BinOper* bin_oper,
                                            llvm::Value* lhs_lv,
                                            llvm::Value* rhs_lv,
                                            const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto lhs = bin_oper->leftOperand();
  const auto rhs = bin_oper->rightOperand();
  const auto& lhs_type = lhs->type();
  const auto& rhs_type = rhs->type();
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  const auto& oper_type = rhs_type->isInterval() ? rhs_type : lhs_type;
  switch (bin_oper->opType()) {
    case hdk::ir::OpType::kMinus:
      return codegenSub(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case hdk::ir::OpType::kPlus:
      return codegenAdd(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case hdk::ir::OpType::kMul:
      return codegenMul(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case hdk::ir::OpType::kDiv:
      return codegenDiv(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type);
    case hdk::ir::OpType::kMod:
      return codegenMod(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

// Handle floating point operand types.
llvm::Value* CodeGenerator::codegenFpArith(const hdk::ir::BinOper* bin_oper,
                                           llvm::Value* lhs_lv,
                                           llvm::Value* rhs_lv) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto lhs = bin_oper->leftOperand();
  const auto rhs = bin_oper->rightOperand();
  const auto& lhs_type = lhs->type();
  const auto& rhs_type = rhs->type();
  const auto fp_typename = numeric_type_name(lhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  llvm::ConstantFP* fp_null{lhs_type->isFp32() ? cgen_state_->llFp(NULL_FLOAT)
                                               : cgen_state_->llFp(NULL_DOUBLE)};
  switch (bin_oper->opType()) {
    case hdk::ir::OpType::kMinus:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFSub(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("sub_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case hdk::ir::OpType::kPlus:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFAdd(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("add_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case hdk::ir::OpType::kMul:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFMul(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("mul_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case hdk::ir::OpType::kDiv:
      return codegenDiv(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : fp_typename,
                        null_check_suffix,
                        lhs_type);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

namespace {

bool is_temporary_column(const hdk::ir::Expr* expr) {
  const auto col_expr = dynamic_cast<const hdk::ir::ColumnVar*>(expr);
  if (!col_expr) {
    return false;
  }
  return col_expr->tableId() < 0;
}

}  // namespace

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool CodeGenerator::checkExpressionRanges(const hdk::ir::BinOper* bin_oper,
                                          int64_t min,
                                          int64_t max) {
  if (is_temporary_column(bin_oper->leftOperand()) ||
      is_temporary_column(bin_oper->rightOperand())) {
    // Computing the range for temporary columns is a lot more expensive than the overflow
    // check.
    return false;
  }
  if (bin_oper->type()->isDecimal()) {
    return false;
  }

  CHECK(plan_state_);
  if (executor_) {
    auto expr_range_info =
        plan_state_->query_infos_.size() > 0
            ? getExpressionRange(bin_oper, plan_state_->query_infos_, executor())
            : ExpressionRange::makeInvalidRange();
    if (expr_range_info.getType() != ExpressionRangeType::Integer) {
      return false;
    }
    if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max) {
      return true;
    }
  }

  return false;
}

llvm::Value* CodeGenerator::codegenAdd(const hdk::ir::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const hdk::ir::Type* type,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(type->isInteger() || type->isDecimal() || type->isInterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(type->size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, type);
  }

  llvm::BasicBlock* add_ok{nullptr};
  llvm::BasicBlock* add_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    add_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "add_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, add_ok, type);
    }
    add_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "add_fail", cgen_state_->current_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSGT(
            rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_max, lhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSLT(
            rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_min, lhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, add_fail, add_ok);
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  auto ret = null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "add_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_value(type))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(add_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenSub(const hdk::ir::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const hdk::ir::Type* type,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(type->isInteger() || type->isDecimal() || type->isInterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(type->size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, type);
  }

  llvm::BasicBlock* sub_ok{nullptr};
  llvm::BasicBlock* sub_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    sub_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "sub_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, sub_ok, type);
    }
    sub_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "sub_fail", cgen_state_->current_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(
            rhs_lv, const_zero),  // sub going up, check the max
        cgen_state_->ir_builder_.CreateICmpSGT(
            lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_max, rhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(
            rhs_lv, const_zero),  // sub going down, check the min
        cgen_state_->ir_builder_.CreateICmpSLT(
            lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_min, rhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, sub_fail, sub_ok);
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  auto ret = null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "sub_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_value(type))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(sub_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  return ret;
}

void CodeGenerator::codegenSkipOverflowCheckForNull(llvm::Value* lhs_lv,
                                                    llvm::Value* rhs_lv,
                                                    llvm::BasicBlock* no_overflow_bb,
                                                    const hdk::ir::Type* type) {
  const auto lhs_is_null_lv = codegenIsNullNumber(lhs_lv, type);
  const auto has_null_operand_lv =
      rhs_lv ? cgen_state_->ir_builder_.CreateOr(lhs_is_null_lv,
                                                 codegenIsNullNumber(rhs_lv, type))
             : lhs_is_null_lv;
  auto operands_not_null = llvm::BasicBlock::Create(
      cgen_state_->context_, "operands_not_null", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(
      has_null_operand_lv, no_overflow_bb, operands_not_null);
  cgen_state_->ir_builder_.SetInsertPoint(operands_not_null);
}

llvm::Value* CodeGenerator::codegenMul(const hdk::ir::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const hdk::ir::Type* type,
                                       const CompilationOptions& co,
                                       bool downscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(type->isInteger() || type->isDecimal() || type->isInterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(type->size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, type);
  }

  llvm::BasicBlock* mul_ok{nullptr};
  llvm::BasicBlock* mul_fail{nullptr};

  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    mul_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, mul_ok, type);
    }

    // Overflow check following LLVM implementation
    // https://github.com/hdoc/llvm-project/blob/release/15.x//llvm/include/llvm/Support/MathExtras.h

    // Create LLVM Basic Block for control flow
    mul_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_fail", cgen_state_->current_func_);
    auto mul_check_1 = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_check_1", cgen_state_->current_func_);
    auto mul_check_2 = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_check_2", cgen_state_->current_func_);

    // Define LLVM constant
    auto const_zero = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
    auto const_one = llvm::ConstantInt::get(rhs_lv->getType(), 1, true);

    // If any of the args was 0, no overflow occurs
    auto lhs_is_zero = cgen_state_->ir_builder_.CreateICmpEQ(lhs_lv, const_zero);
    auto rhs_is_zero = cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, const_zero);
    auto args_zero = cgen_state_->ir_builder_.CreateOr(rhs_is_zero, lhs_is_zero);
    cgen_state_->ir_builder_.CreateCondBr(args_zero, mul_ok, mul_check_1);
    cgen_state_->ir_builder_.SetInsertPoint(mul_check_1);

    // Check the sign of the args
    auto lhs_is_neg = cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero);
    auto rhs_is_neg = cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, const_zero);
    auto args_is_neg = cgen_state_->ir_builder_.CreateOr(lhs_is_neg, rhs_is_neg);

    auto lhs_is_pos = cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero);
    auto rhs_is_pos = cgen_state_->ir_builder_.CreateICmpSGT(rhs_lv, const_zero);
    auto args_is_pos = cgen_state_->ir_builder_.CreateOr(lhs_is_pos, rhs_is_pos);

    // Get the absolute value of the args
    auto lhs_neg = cgen_state_->ir_builder_.CreateNeg(lhs_lv);
    auto rhs_neg = cgen_state_->ir_builder_.CreateNeg(rhs_lv);
    auto lhs_pos = cgen_state_->ir_builder_.CreateSelect(lhs_is_neg, lhs_neg, lhs_lv);
    auto rhs_pos = cgen_state_->ir_builder_.CreateSelect(rhs_is_neg, rhs_neg, rhs_lv);

    // lhs and rhs are in [1, 2^n], where n is the number of digits.
    // Check how the max allowed absolute value (2^n for negative, 2^(n-1) for
    // positive) divided by an argument compares to the other.

    // if(IsNegative) && UX > (static_cast<U>(std::numeric_limits<T>::max()) + U(1)) / UY;
    auto max_plus_one = cgen_state_->ir_builder_.CreateAdd(chosen_max, const_one);
    auto div_limit_plus = cgen_state_->ir_builder_.CreateUDiv(max_plus_one, rhs_pos);
    auto cmp_plus = cgen_state_->ir_builder_.CreateICmpUGT(lhs_pos, div_limit_plus);
    auto neg_overflow = cgen_state_->ir_builder_.CreateAnd(args_is_neg, cmp_plus);
    cgen_state_->ir_builder_.CreateCondBr(neg_overflow, mul_fail, mul_check_2);
    cgen_state_->ir_builder_.SetInsertPoint(mul_check_2);

    // if !(IsNegative) && UX > (static_cast<U>(std::numeric_limits<T>::max())) / UY;
    auto div_limit = cgen_state_->ir_builder_.CreateUDiv(chosen_max, rhs_pos);
    auto cmp = cgen_state_->ir_builder_.CreateICmpUGT(lhs_pos, div_limit);
    auto pos_overflow = cgen_state_->ir_builder_.CreateAnd(args_is_pos, cmp);
    cgen_state_->ir_builder_.CreateCondBr(pos_overflow, mul_fail, mul_ok);
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }

  const auto ret =
      null_check_suffix.empty()
          ? cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv)
          : cgen_state_->emitCall(
                "mul_" + null_typename + null_check_suffix,
                {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_value(type))});

  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(mul_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenDiv(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const hdk::ir::Type* type,
                                       bool upscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  if (type->isDecimal()) {
    auto scale = type->as<hdk::ir::DecimalType>()->scale();
    if (upscale) {
      CHECK(lhs_lv->getType()->isIntegerTy());
      const auto scale_lv =
          llvm::ConstantInt::get(lhs_lv->getType(), exp_to_scale(scale));

      lhs_lv = cgen_state_->ir_builder_.CreateSExt(
          lhs_lv, get_int_type(64, cgen_state_->context_));
      llvm::Value* chosen_max{nullptr};
      llvm::Value* chosen_min{nullptr};
      std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(8, true);
      auto decimal_div_ok = llvm::BasicBlock::Create(
          cgen_state_->context_, "decimal_div_ok", cgen_state_->current_func_);
      if (!null_check_suffix.empty()) {
        codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, decimal_div_ok, type);
      }
      auto decimal_div_fail = llvm::BasicBlock::Create(
          cgen_state_->context_, "decimal_div_fail", cgen_state_->current_func_);
      auto lhs_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() /
                     exp_to_scale(scale);
      auto lhs_max_lv =
          llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), lhs_max);
      llvm::Value* detected{nullptr};
      if (!type->nullable()) {
        detected = cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, lhs_max_lv);
      } else {
        detected = toBool(
            cgen_state_->emitCall("gt_" + numeric_type_name(type) + "_nullable",
                                  {lhs_lv,
                                   lhs_max_lv,
                                   cgen_state_->llInt(inline_int_null_value(type)),
                                   cgen_state_->inlineIntNull(type->ctx().boolean())}));
      }
      cgen_state_->ir_builder_.CreateCondBr(detected, decimal_div_fail, decimal_div_ok);

      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_fail);
      cgen_state_->ir_builder_.CreateRet(
          cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_ok);

      lhs_lv =
          null_typename.empty()
              ? cgen_state_->ir_builder_.CreateMul(lhs_lv, scale_lv)
              : cgen_state_->emitCall(
                    "mul_" + numeric_type_name(type) + null_check_suffix,
                    {lhs_lv, scale_lv, cgen_state_->llInt(inline_int_null_value(type))});
    }
  }
  if (config_.exec.codegen.inf_div_by_zero && type->isFloatingPoint()) {
    llvm::Value* inf_lv =
        type->size() == 4 ? cgen_state_->llFp(INF_FLOAT) : cgen_state_->llFp(INF_DOUBLE);
    llvm::Value* null_lv = type->size() == 4 ? cgen_state_->llFp(NULL_FLOAT)
                                             : cgen_state_->llFp(NULL_DOUBLE);
    return cgen_state_->emitCall("safe_inf_div_" + numeric_type_name(type),
                                 {lhs_lv, rhs_lv, inf_lv, null_lv});
  }
  if (config_.exec.codegen.null_div_by_zero) {
    llvm::Value* null_lv{nullptr};
    if (type->isFloatingPoint()) {
      null_lv = type->size() == 4 ? cgen_state_->llFp(NULL_FLOAT)
                                  : cgen_state_->llFp(NULL_DOUBLE);
    } else {
      null_lv = cgen_state_->llInt(inline_int_null_value(type));
    }
    return cgen_state_->emitCall("safe_div_" + numeric_type_name(type),
                                 {lhs_lv, rhs_lv, null_lv});
  }
  cgen_state_->needs_error_check_ = true;
  auto div_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "div_ok", cgen_state_->current_func_);
  if (!null_check_suffix.empty()) {
    codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, div_ok, type);
  }
  auto div_zero = llvm::BasicBlock::Create(
      cgen_state_->context_, "div_zero", cgen_state_->current_func_);
  auto zero_const = rhs_lv->getType()->isIntegerTy()
                        ? llvm::ConstantInt::get(rhs_lv->getType(), 0, true)
                        : llvm::ConstantFP::get(rhs_lv->getType(), 0.);
  cgen_state_->ir_builder_.CreateCondBr(
      zero_const->getType()->isFloatingPointTy()
          ? cgen_state_->ir_builder_.CreateFCmp(
                llvm::FCmpInst::FCMP_ONE, rhs_lv, zero_const)
          : cgen_state_->ir_builder_.CreateICmp(
                llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      div_ok,
      div_zero);
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  auto ret =
      zero_const->getType()->isIntegerTy()
          ? (null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateSDiv(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "div_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_value(type))}))
          : (null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateFDiv(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "div_" + null_typename + null_check_suffix,
                       {lhs_lv,
                        rhs_lv,
                        type->size() == 4 ? cgen_state_->llFp(NULL_FLOAT)
                                          : cgen_state_->llFp(NULL_DOUBLE)}));
  cgen_state_->ir_builder_.SetInsertPoint(div_zero);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(Executor::ERR_DIV_BY_ZERO));
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  return ret;
}

// Handle decimal division by an integer (constant or cast), return null if
// the expression doesn't match this pattern and let the general method kick in.
// For said patterns, we can simply divide the decimal operand by the non-scaled
// integer value instead of using the scaled value preceded by a multiplication.
// It is both more efficient and avoids the overflow for a lot of practical cases.
llvm::Value* CodeGenerator::codegenDeciDiv(const hdk::ir::BinOper* bin_oper,
                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto lhs = bin_oper->leftOperand();
  auto rhs = bin_oper->rightOperand();
  const auto& lhs_type = lhs->type();
  const auto& rhs_type = rhs->type();
  CHECK(lhs_type->isDecimal() && rhs_type->isDecimal());
  auto scale = lhs_type->as<hdk::ir::DecimalType>()->scale();
  CHECK_EQ(scale, rhs_type->as<hdk::ir::DecimalType>()->scale());

  auto rhs_constant = dynamic_cast<const hdk::ir::Constant*>(rhs);
  auto rhs_cast = dynamic_cast<const hdk::ir::UOper*>(rhs);
  if (rhs_constant && !rhs_constant->isNull() && rhs_constant->value().bigintval != 0LL &&
      (rhs_constant->value().bigintval % exp_to_scale(scale)) == 0LL) {
    // can safely downscale a scaled constant
  } else if (rhs_cast && rhs_cast->isCast() && rhs_cast->operand()->type()->isInteger()) {
    // can skip upscale in the int to dec cast
  } else {
    return nullptr;
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  llvm::Value* rhs_lv{nullptr};
  if (rhs_constant) {
    const auto rhs_lit =
        Analyzer::analyzeIntValue(rhs_constant->value().bigintval / exp_to_scale(scale));
    auto rhs_lit_lv = CodeGenerator::codegenIntConst(
        dynamic_cast<const hdk::ir::Constant*>(rhs_lit.get()), cgen_state_);
    rhs_lv = codegenCastBetweenIntTypes(
        rhs_lit_lv, rhs_lit->type(), lhs_type, /*upscale*/ false);
  } else if (rhs_cast) {
    auto rhs_cast_oper = rhs_cast->operand();
    const auto& rhs_cast_oper_type = rhs_cast_oper->type();
    auto rhs_cast_oper_lv = codegen(rhs_cast_oper, true, co).front();
    rhs_lv = codegenCastBetweenIntTypes(
        rhs_cast_oper_lv, rhs_cast_oper_type, lhs_type, /*upscale*/ false);
  } else {
    CHECK(false);
  }
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  return codegenDiv(lhs_lv,
                    rhs_lv,
                    null_check_suffix.empty() ? "" : int_typename,
                    null_check_suffix,
                    lhs_type,
                    /*upscale*/ false);
}

llvm::Value* CodeGenerator::codegenMod(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const hdk::ir::Type* type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(type->isInteger());
  cgen_state_->needs_error_check_ = true;
  // Generate control flow for division by zero error handling.
  auto mod_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "mod_ok", cgen_state_->current_func_);
  auto mod_zero = llvm::BasicBlock::Create(
      cgen_state_->context_, "mod_zero", cgen_state_->current_func_);
  auto zero_const = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
  cgen_state_->ir_builder_.CreateCondBr(
      cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      mod_ok,
      mod_zero);
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  auto ret = null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateSRem(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "mod_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_value(type))});
  if (config_.exec.codegen.null_mod_by_zero) {
    auto mod_res = llvm::BasicBlock::Create(
        cgen_state_->context_, "mod_res", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateBr(mod_res);
    cgen_state_->ir_builder_.SetInsertPoint(mod_zero);
    cgen_state_->ir_builder_.CreateBr(mod_res);
    cgen_state_->ir_builder_.SetInsertPoint(mod_res);
    auto phi = cgen_state_->ir_builder_.CreatePHI(ret->getType(), 2);
    phi->addIncoming(ret, mod_ok);
    phi->addIncoming(llvm::ConstantInt::get(ret->getType(), inline_int_null_value(type)),
                     mod_zero);
    ret = phi;
  } else {
    cgen_state_->ir_builder_.SetInsertPoint(mod_zero);
    cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(Executor::ERR_DIV_BY_ZERO));
    cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  }
  return ret;
}

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool CodeGenerator::checkExpressionRanges(const hdk::ir::UOper* uoper,
                                          int64_t min,
                                          int64_t max) {
  if (uoper->type()->isDecimal()) {
    return false;
  }

  CHECK(plan_state_);
  if (executor_) {
    auto expr_range_info =
        plan_state_->query_infos_.size() > 0
            ? getExpressionRange(uoper, plan_state_->query_infos_, executor())
            : ExpressionRange::makeInvalidRange();
    if (expr_range_info.getType() != ExpressionRangeType::Integer) {
      return false;
    }
    if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max) {
      return true;
    }
  }

  return false;
}

llvm::Value* CodeGenerator::codegenUMinus(const hdk::ir::UOper* uoper,
                                          const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(uoper->isUMinus());
  const auto operand_lv = codegen(uoper->operand(), true, co).front();
  const auto& type = uoper->type();
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  bool need_overflow_check = false;
  if (type->isInteger() || type->isDecimal() || type->isInterval()) {
    std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(type->size(), true);
    need_overflow_check = !checkExpressionRanges(
        uoper,
        static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
        static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  }
  llvm::BasicBlock* uminus_ok{nullptr};
  llvm::BasicBlock* uminus_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    uminus_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "uminus_ok", cgen_state_->current_func_);
    if (type->nullable()) {
      codegenSkipOverflowCheckForNull(operand_lv, nullptr, uminus_ok, type);
    }
    uminus_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "uminus_fail", cgen_state_->current_func_);
    auto const_min = llvm::ConstantInt::get(
        operand_lv->getType(),
        static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
        true);
    auto overflow = cgen_state_->ir_builder_.CreateICmpEQ(operand_lv, const_min);
    cgen_state_->ir_builder_.CreateCondBr(overflow, uminus_fail, uminus_ok);
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  auto ret =
      !type->nullable()
          ? (type->isFloatingPoint() ? cgen_state_->ir_builder_.CreateFNeg(operand_lv)
                                     : cgen_state_->ir_builder_.CreateNeg(operand_lv))
          : cgen_state_->emitCall(
                "uminus_" + numeric_type_name(type) + "_nullable",
                {operand_lv, static_cast<llvm::Value*>(cgen_state_->inlineNull(type))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(uminus_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  return ret;
}

llvm::Function* CodeGenerator::getArithWithOverflowIntrinsic(
    const hdk::ir::BinOper* bin_oper,
    llvm::Type* type) {
  llvm::Intrinsic::ID fn_id{llvm::Intrinsic::not_intrinsic};
  switch (bin_oper->opType()) {
    case hdk::ir::OpType::kMinus:
      fn_id = llvm::Intrinsic::ssub_with_overflow;
      break;
    case hdk::ir::OpType::kPlus:
      fn_id = llvm::Intrinsic::sadd_with_overflow;
      break;
    case hdk::ir::OpType::kMul:
      fn_id = llvm::Intrinsic::smul_with_overflow;
      break;
    default:
      LOG(FATAL) << "unexpected arith with overflow optype: " << bin_oper->toString();
  }

  return llvm::Intrinsic::getDeclaration(cgen_state_->module_, fn_id, type);
}

llvm::Value* CodeGenerator::codegenBinOpWithOverflowForCPU(
    const hdk::ir::BinOper* bin_oper,
    llvm::Value* lhs_lv,
    llvm::Value* rhs_lv,
    const std::string& null_check_suffix,
    const hdk::ir::Type* type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  cgen_state_->needs_error_check_ = true;

  llvm::BasicBlock* check_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "ovf_ok", cgen_state_->current_func_);
  llvm::BasicBlock* check_fail = llvm::BasicBlock::Create(
      cgen_state_->context_, "ovf_detected", cgen_state_->current_func_);
  llvm::BasicBlock* null_check{nullptr};

  if (!null_check_suffix.empty()) {
    null_check = cgen_state_->ir_builder_.GetInsertBlock();
    codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, check_ok, type);
  }

  // Compute result and overflow flag
  auto func = getArithWithOverflowIntrinsic(bin_oper, lhs_lv->getType());
  auto ret_and_overflow = cgen_state_->ir_builder_.CreateCall(
      func, std::vector<llvm::Value*>{lhs_lv, rhs_lv});
  auto ret = cgen_state_->ir_builder_.CreateExtractValue(ret_and_overflow,
                                                         std::vector<unsigned>{0});
  auto overflow = cgen_state_->ir_builder_.CreateExtractValue(ret_and_overflow,
                                                              std::vector<unsigned>{1});
  auto val_bb = cgen_state_->ir_builder_.GetInsertBlock();

  // Return error on overflow
  cgen_state_->ir_builder_.CreateCondBr(overflow, check_fail, check_ok);
  cgen_state_->ir_builder_.SetInsertPoint(check_fail);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

  cgen_state_->ir_builder_.SetInsertPoint(check_ok);

  // In case of null check we have to use NULL result on check fail
  if (null_check) {
    auto phi = cgen_state_->ir_builder_.CreatePHI(ret->getType(), 2);
    phi->addIncoming(llvm::ConstantInt::get(ret->getType(), inline_int_null_value(type)),
                     null_check);
    phi->addIncoming(ret, val_bb);
    ret = phi;
  }

  return ret;
}
