/*
 * Copyright 2018 OmniSci, Inc.
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
#include "WindowContext.h"

llvm::Value* Executor::codegenWindowFunction(const size_t target_index,
                                             const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  CodeGenerator code_generator(this, co.codegen_traits_desc);
  const auto window_func_context =
      WindowProjectNodeContext::get(this)->activateWindowFunctionContext(this,
                                                                         target_index);
  const auto window_func = window_func_context->getWindowFunction();
  switch (window_func->kind()) {
    case hdk::ir::WindowFunctionKind::RowNumber:
    case hdk::ir::WindowFunctionKind::Rank:
    case hdk::ir::WindowFunctionKind::DenseRank:
    case hdk::ir::WindowFunctionKind::NTile: {
      return cgen_state_->emitCall("row_number_window_func",
                                   {cgen_state_->llInt(reinterpret_cast<const int64_t>(
                                        window_func_context->output())),
                                    code_generator.posArg(nullptr)});
    }
    case hdk::ir::WindowFunctionKind::PercentRank:
    case hdk::ir::WindowFunctionKind::CumeDist: {
      return cgen_state_->emitCall("percent_window_func",
                                   {cgen_state_->llInt(reinterpret_cast<const int64_t>(
                                        window_func_context->output())),
                                    code_generator.posArg(nullptr)});
    }
    case hdk::ir::WindowFunctionKind::Lag:
    case hdk::ir::WindowFunctionKind::Lead:
    case hdk::ir::WindowFunctionKind::FirstValue:
    case hdk::ir::WindowFunctionKind::LastValue: {
      CHECK(WindowProjectNodeContext::get(this));
      const auto& args = window_func->args();
      CHECK(!args.empty());
      const auto arg_lvs = code_generator.codegen(args.front().get(), true, co);
      CHECK_EQ(arg_lvs.size(), size_t(1));
      return arg_lvs.front();
    }
    case hdk::ir::WindowFunctionKind::Avg:
    case hdk::ir::WindowFunctionKind::Min:
    case hdk::ir::WindowFunctionKind::Max:
    case hdk::ir::WindowFunctionKind::Sum:
    case hdk::ir::WindowFunctionKind::Count: {
      return codegenWindowFunctionAggregate(co);
    }
    default: {
      LOG(FATAL) << "Invalid window function kind";
    }
  }
  return nullptr;
}

namespace {

std::string get_window_agg_name(const hdk::ir::WindowFunctionKind kind,
                                const hdk::ir::Type* window_func_type) {
  std::string agg_name;
  switch (kind) {
    case hdk::ir::WindowFunctionKind::Min: {
      agg_name = "agg_min";
      break;
    }
    case hdk::ir::WindowFunctionKind::Max: {
      agg_name = "agg_max";
      break;
    }
    case hdk::ir::WindowFunctionKind::Avg:
    case hdk::ir::WindowFunctionKind::Sum: {
      agg_name = "agg_sum";
      break;
    }
    case hdk::ir::WindowFunctionKind::Count: {
      agg_name = "agg_count";
      break;
    }
    default: {
      LOG(FATAL) << "Invalid window function kind";
    }
  }
  if (window_func_type->isFp32()) {
    agg_name += "_float";
  } else if (window_func_type->isFp64()) {
    agg_name += "_double";
  }
  return agg_name;
}

const hdk::ir::Type* get_adjusted_window_type(
    const hdk::ir::WindowFunction* window_func) {
  const auto& args = window_func->args();
  return ((window_func->kind() == hdk::ir::WindowFunctionKind::Count && !args.empty()) ||
          window_func->kind() == hdk::ir::WindowFunctionKind::Avg)
             ? args.front()->type()
             : window_func->type();
}

}  // namespace

llvm::Value* Executor::aggregateWindowStatePtr(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto arg_type = get_adjusted_window_type(window_func);
  llvm::Type* aggregate_state_type =
      arg_type->isFp32() 
          ? cgen_traits.localPointerType(get_int_type(32, cgen_state_->context_))
          : cgen_traits.localPointerType(get_int_type(64, cgen_state_->context_));
  const auto aggregate_state_i64 = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->aggregateState()));
  return cgen_state_->ir_builder_.CreateIntToPtr(aggregate_state_i64,
                                                 aggregate_state_type);
}

llvm::Value* Executor::codegenWindowFunctionAggregate(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto reset_state_false_bb = codegenWindowResetStateControlFlow(co);
  auto aggregate_state = aggregateWindowStatePtr(co);
  llvm::Value* aggregate_state_count = nullptr;
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  if (window_func->kind() == hdk::ir::WindowFunctionKind::Avg) {
    const auto aggregate_state_count_i64 = cgen_state_->llInt(
        reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
    const auto pi64_type = cgen_traits.localPointerType(get_int_type(64, cgen_state_->context_));
    aggregate_state_count =
        cgen_state_->ir_builder_.CreateIntToPtr(aggregate_state_count_i64, pi64_type);
  }
  codegenWindowFunctionStateInit(aggregate_state, co);
  if (window_func->kind() == hdk::ir::WindowFunctionKind::Avg) {
    const auto count_zero = cgen_state_->llInt(int64_t(0));
    cgen_state_->emitCall("agg_id", {aggregate_state_count, count_zero});
  }
  cgen_state_->ir_builder_.CreateBr(reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_false_bb);
  CHECK(WindowProjectNodeContext::get(this));
  return codegenWindowFunctionAggregateCalls(aggregate_state, co);
}

llvm::BasicBlock* Executor::codegenWindowResetStateControlFlow(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto bitset = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->partitionStart()));
  const auto min_val = cgen_state_->llInt(int64_t(0));
  const auto max_val = cgen_state_->llInt(window_func_context->elementCount() - 1);
  const auto null_val = cgen_state_->llInt(inline_int_null_value<int64_t>());
  const auto null_bool_val = cgen_state_->llInt<int8_t>(inline_int_null_value<int8_t>());
  CodeGenerator code_generator(this, co.codegen_traits_desc);
  const auto reset_state =
      code_generator.toBool(cgen_state_->emitCall("bit_is_set",
                                                  {bitset,
                                                   code_generator.posArg(nullptr),
                                                   min_val,
                                                   max_val,
                                                   null_val,
                                                   null_bool_val}));
  const auto reset_state_true_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.true", cgen_state_->current_func_);
  const auto reset_state_false_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.false", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(
      reset_state, reset_state_true_bb, reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_true_bb);
  return reset_state_false_bb;
}

void Executor::codegenWindowFunctionStateInit(llvm::Value* aggregate_state, const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_type = get_adjusted_window_type(window_func);
  const auto window_func_null_val =
      window_func_type->isFloatingPoint()
          ? cgen_state_->inlineFpNull(window_func_type)
          : cgen_state_->castToTypeIn(cgen_state_->inlineIntNull(window_func_type), 64);
  llvm::Value* window_func_init_val;
  if (window_func_context->getWindowFunction()->kind() ==
      hdk::ir::WindowFunctionKind::Count) {
    if (window_func_type->isFp32()) {
      window_func_init_val = cgen_state_->llFp(float(0));
    } else if (window_func_type->isFp64()) {
      window_func_init_val = cgen_state_->llFp(double(0));
    } else {
      window_func_init_val = cgen_state_->llInt(int64_t(0));
    }
  } else {
    window_func_init_val = window_func_null_val;
  }
  const auto pi32_type = cgen_traits.localPointerType(get_int_type(32, cgen_state_->context_));
  if (window_func_type->isFp64()) {
    cgen_state_->emitCall("agg_id_double", {aggregate_state, window_func_init_val});
  } else if (window_func_type->isFp32()) {
    aggregate_state = cgen_state_->ir_builder_.CreateBitCast(aggregate_state, pi32_type);
    cgen_state_->emitCall("agg_id_float", {aggregate_state, window_func_init_val});
  } else {
    cgen_state_->emitCall("agg_id", {aggregate_state, window_func_init_val});
  }
}

llvm::Value* Executor::codegenWindowFunctionAggregateCalls(llvm::Value* aggregate_state,
                                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_type = get_adjusted_window_type(window_func);
  const auto window_func_null_val =
      window_func_type->isFloatingPoint()
          ? cgen_state_->inlineFpNull(window_func_type)
          : cgen_state_->castToTypeIn(cgen_state_->inlineIntNull(window_func_type), 64);
  const auto& args = window_func->args();
  llvm::Value* crt_val;
  if (args.empty()) {
    CHECK(window_func->kind() == hdk::ir::WindowFunctionKind::Count);
    crt_val = cgen_state_->llInt(int64_t(1));
  } else {
    CodeGenerator code_generator(this, co.codegen_traits_desc);
    const auto arg_lvs = code_generator.codegen(args.front().get(), true, co);
    CHECK_EQ(arg_lvs.size(), size_t(1));
    if (window_func->kind() == hdk::ir::WindowFunctionKind::Sum &&
        !window_func_type->isFloatingPoint()) {
      crt_val = code_generator.codegenCastBetweenIntTypes(
          arg_lvs.front(), args.front()->type(), window_func_type, false);
    } else {
      crt_val = window_func_type->isFp32()
                    ? arg_lvs.front()
                    : cgen_state_->castToTypeIn(arg_lvs.front(), 64);
    }
  }
  const auto agg_name = get_window_agg_name(window_func->kind(), window_func_type);
  llvm::Value* multiplicity_lv = nullptr;
  if (args.empty()) {
    cgen_state_->emitCall(agg_name, {aggregate_state, crt_val});
  } else {
    cgen_state_->emitCall(agg_name + "_skip_val",
                          {aggregate_state, crt_val, window_func_null_val});
  }
  if (window_func->kind() == hdk::ir::WindowFunctionKind::Avg) {
    codegenWindowAvgEpilogue(crt_val, window_func_null_val, multiplicity_lv, co);
  }
  return codegenAggregateWindowState(co);
}

void Executor::codegenWindowAvgEpilogue(llvm::Value* crt_val,
                                        llvm::Value* window_func_null_val,
                                        llvm::Value* multiplicity_lv,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_type = get_adjusted_window_type(window_func);
  const auto pi32_type = cgen_traits.localPointerType(get_int_type(32, cgen_state_->context_));
  const auto pi64_type = cgen_traits.localPointerType(get_int_type(64, cgen_state_->context_));
  const auto aggregate_state_type = window_func_type->isFp32() ? pi32_type : pi64_type;
  const auto aggregate_state_count_i64 = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
  auto aggregate_state_count = cgen_state_->ir_builder_.CreateIntToPtr(
      aggregate_state_count_i64, aggregate_state_type);
  std::string agg_count_func_name = "agg_count";
  if (window_func_type->isFp32()) {
    agg_count_func_name += "_float";
  } else if (window_func_type->isFp64()) {
    agg_count_func_name += "_double";
  }
  agg_count_func_name += "_skip_val";
  cgen_state_->emitCall(agg_count_func_name,
                        {aggregate_state_count, crt_val, window_func_null_val});
}

llvm::Value* Executor::codegenAggregateWindowState(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto pi32_type = cgen_traits.localPointerType(get_int_type(32, cgen_state_->context_));
  const auto pi64_type = cgen_traits.localPointerType(get_int_type(64, cgen_state_->context_));
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const hdk::ir::WindowFunction* window_func = window_func_context->getWindowFunction();
  const auto window_func_type = get_adjusted_window_type(window_func);
  const auto aggregate_state_type = window_func_type->isFp32() ? pi32_type : pi64_type;
  auto aggregate_state = aggregateWindowStatePtr(co);
  if (window_func->kind() == hdk::ir::WindowFunctionKind::Avg) {
    const auto aggregate_state_count_i64 = cgen_state_->llInt(
        reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
    auto aggregate_state_count = cgen_state_->ir_builder_.CreateIntToPtr(
        aggregate_state_count_i64, aggregate_state_type);
    const auto double_null_lv = cgen_state_->inlineFpNull(window_func_type->ctx().fp64());
    if (window_func_type->isFp32()) {
      return cgen_state_->emitCall(
          "load_avg_float", {aggregate_state, aggregate_state_count, double_null_lv});
    } else if (window_func_type->isFp64()) {
      return cgen_state_->emitCall(
          "load_avg_double", {aggregate_state, aggregate_state_count, double_null_lv});
    } else if (window_func_type->isDecimal()) {
      return cgen_state_->emitCall(
          "load_avg_decimal",
          {aggregate_state,
           aggregate_state_count,
           double_null_lv,
           cgen_state_->llInt<int32_t>(
               window_func_type->as<hdk::ir::DecimalType>()->scale())});
    } else {
      return cgen_state_->emitCall(
          "load_avg_int", {aggregate_state, aggregate_state_count, double_null_lv});
    }
  }
  if (window_func->kind() == hdk::ir::WindowFunctionKind::Count) {
    return cgen_state_->ir_builder_.CreateLoad(
        aggregate_state->getType()->getPointerElementType(), aggregate_state);
  }
  if (window_func_type->isFp32()) {
    return cgen_state_->emitCall("load_float", {aggregate_state});
  } else if (window_func_type->isFp64()) {
    return cgen_state_->emitCall("load_double", {aggregate_state});
  } else {
    return cgen_state_->ir_builder_.CreateLoad(
        aggregate_state->getType()->getPointerElementType(), aggregate_state);
  }
}
