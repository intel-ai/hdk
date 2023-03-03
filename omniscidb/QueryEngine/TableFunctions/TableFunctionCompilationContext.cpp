/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_os_ostream.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Compiler/Backend.h"
#include "QueryEngine/Compiler/HelperFunctions.h"

namespace {

llvm::Function* generate_entry_point(const CgenState* cgen_state,
                                     const CompilationOptions& co) {
  auto& ctx = cgen_state->context_;
  compiler::CodegenTraits cgen_traits =
      compiler::CodegenTraits::get(co.codegen_traits_desc);
  const auto pi8_type = cgen_traits.localPointerType(get_int_type(8, ctx));
  const auto ppi8_type = cgen_traits.localPointerType(pi8_type);
  const auto pi64_type = cgen_traits.localPointerType(get_int_type(64, ctx));
  const auto ppi64_type = cgen_traits.localPointerType(pi64_type);
  const auto i32_type = get_int_type(32, ctx);

  const auto func_type = llvm::FunctionType::get(
      i32_type, {pi8_type, ppi8_type, pi64_type, ppi64_type, pi64_type}, false);

  auto func = llvm::Function::Create(func_type,
                                     llvm::Function::ExternalLinkage,
                                     "call_table_function",
                                     cgen_state->module_);
  auto arg_it = func->arg_begin();
  const auto mgr_arg = &*arg_it;
  mgr_arg->setName("mgr_ptr");
  const auto input_cols_arg = &*(++arg_it);
  input_cols_arg->setName("input_col_buffers");
  const auto input_row_counts = &*(++arg_it);
  input_row_counts->setName("input_row_counts");
  const auto output_buffers = &*(++arg_it);
  output_buffers->setName("output_buffers");
  const auto output_row_count = &*(++arg_it);
  output_row_count->setName("output_row_count");
  return func;
}

inline llvm::Type* get_llvm_type_from_sql_column_type(
    const hdk::ir::Type* elem_type,
    llvm::LLVMContext& ctx,
    const compiler::CodegenTraits& cgen_traits) {
  const unsigned current_addr_space = cgen_traits.getLocalAddrSpace();
  if (elem_type->isFloatingPoint()) {
    return get_fp_ptr_type(elem_type->size() * 8, ctx, current_addr_space);
  } else if (elem_type->isBoolean()) {
    return get_int_ptr_type(8, ctx, current_addr_space);
  } else if (elem_type->isInteger() || elem_type->isExtDictionary()) {
    return get_int_ptr_type(elem_type->size() * 8, ctx, current_addr_space);
  } else if (elem_type->isString()) {
    return get_int_ptr_type(8, ctx, current_addr_space);
  }
  LOG(FATAL) << "get_llvm_type_from_sql_column_type: not implemented for "
             << elem_type->toString();
  return nullptr;
}

std::tuple<llvm::Value*, llvm::Value*> alloc_column(
    std::string col_name,
    const size_t index,
    const hdk::ir::Type* data_target_type,
    llvm::Value* data_ptr,
    llvm::Value* data_size,
    llvm::LLVMContext& ctx,
    llvm::IRBuilder<>& ir_builder,
    compiler::CodegenTraitsDescriptor codegen_traits_desc) {
  /*
    Creates a new Column instance of given element type and initialize
    its data ptr and sz members when specified. If data ptr or sz are
    unspecified (have nullptr values) then the corresponding members
    are initialized with NULL and -1, respectively.

    Return a pair of Column allocation (caller should apply
    builder.CreateLoad to it in order to construct a Column instance
    as a value) and a pointer to the Column instance.
   */
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(codegen_traits_desc);
  llvm::Type* data_ptr_llvm_type =
      get_llvm_type_from_sql_column_type(data_target_type, ctx, cgen_traits);
  llvm::StructType* col_struct_type =
      llvm::StructType::get(ctx,
                            {
                                data_ptr_llvm_type,         /* T* ptr */
                                llvm::Type::getInt64Ty(ctx) /* int64_t sz */
                            });
  llvm::Value* col = ir_builder.CreateAlloca(col_struct_type);
  if (col->getType()->getPointerAddressSpace() != codegen_traits_desc.local_addr_space_) {
    col = ir_builder.CreateAddrSpaceCast(
        col,
        llvm::PointerType::get(col->getType()->getPointerElementType(),
                               codegen_traits_desc.local_addr_space_),
        "col.cast");
  }
  col->setName(col_name);
  auto col_ptr_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 0);
  auto col_sz_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 1);
  col_ptr_ptr->setName(col_name + ".ptr");
  col_sz_ptr->setName(col_name + ".sz");

  if (data_ptr != nullptr) {
    if (data_ptr->getType() == data_ptr_llvm_type->getPointerElementType()) {
      ir_builder.CreateStore(data_ptr, col_ptr_ptr);
    } else {
      auto tmp = ir_builder.CreateBitCast(data_ptr, data_ptr_llvm_type);
      ir_builder.CreateStore(tmp, col_ptr_ptr);
    }
  } else {
    ir_builder.CreateStore(llvm::Constant::getNullValue(data_ptr_llvm_type), col_ptr_ptr);
  }
  if (data_size != nullptr) {
    auto data_size_type = data_size->getType();
    llvm::Value* size_val = nullptr;
    if (data_size_type->isPointerTy()) {
      CHECK(data_size_type->getPointerElementType()->isIntegerTy(64));
      size_val =
          ir_builder.CreateLoad(data_size->getType()->getPointerElementType(), data_size);
    } else {
      CHECK(data_size_type->isIntegerTy(64));
      size_val = data_size;
    }
    ir_builder.CreateStore(size_val, col_sz_ptr);
  } else {
    auto const_minus1 = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), -1, true);
    ir_builder.CreateStore(const_minus1, col_sz_ptr);
  }
  auto col_ptr = ir_builder.CreatePointerCast(
      col_ptr_ptr,
      llvm::Type::getInt8PtrTy(ctx, col_ptr_ptr->getType()->getPointerAddressSpace()));
  col_ptr->setName(col_name + "_ptr");
  return {col, col_ptr};
}

llvm::Value* alloc_column_list(std::string col_list_name,
                               const hdk::ir::Type* data_target_type,
                               llvm::Value* data_ptrs,
                               int length,
                               llvm::Value* data_size,
                               llvm::LLVMContext& ctx,
                               llvm::IRBuilder<>& ir_builder,
                               compiler::CodegenTraitsDescriptor codegen_traits_desc) {
  /*
    Creates a new ColumnList instance of given element type and initialize
    its members. If data ptr or size are unspecified (have nullptr
    values) then the corresponding members are initialized with NULL
    and -1, respectively.
   */
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(codegen_traits_desc);
  llvm::Type* data_ptrs_llvm_type =
      cgen_traits.localPointerType(llvm::Type::getInt8Ty(ctx));

  llvm::StructType* col_list_struct_type =
      llvm::StructType::get(ctx,
                            {
                                data_ptrs_llvm_type,         /* int8_t* ptrs */
                                llvm::Type::getInt64Ty(ctx), /* int64_t length */
                                llvm::Type::getInt64Ty(ctx)  /* int64_t size */
                            });
  llvm::Value* col_list = ir_builder.CreateAlloca(col_list_struct_type);
  if (col_list->getType()->getPointerAddressSpace() !=
      codegen_traits_desc.local_addr_space_) {
    col_list = ir_builder.CreateAddrSpaceCast(
        col_list,
        llvm::PointerType::get(col_list->getType()->getPointerElementType(),
                               codegen_traits_desc.local_addr_space_),
        "col.list.cast");
  }

  col_list->setName(col_list_name);
  auto col_list_ptr_ptr = ir_builder.CreateStructGEP(col_list_struct_type, col_list, 0);
  auto col_list_length_ptr =
      ir_builder.CreateStructGEP(col_list_struct_type, col_list, 1);
  auto col_list_size_ptr = ir_builder.CreateStructGEP(col_list_struct_type, col_list, 2);

  col_list_ptr_ptr->setName(col_list_name + ".ptrs");
  col_list_length_ptr->setName(col_list_name + ".length");
  col_list_size_ptr->setName(col_list_name + ".size");

  CHECK(length >= 0);
  auto const_length = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), length, true);

  if (data_ptrs != nullptr) {
    if (data_ptrs->getType() == data_ptrs_llvm_type->getPointerElementType()) {
      ir_builder.CreateStore(data_ptrs, col_list_ptr_ptr);
    } else {
      auto tmp = ir_builder.CreateBitCast(data_ptrs, data_ptrs_llvm_type);
      ir_builder.CreateStore(tmp, col_list_ptr_ptr);
    }
  } else {
    ir_builder.CreateStore(llvm::Constant::getNullValue(data_ptrs_llvm_type),
                           col_list_ptr_ptr);
  }

  ir_builder.CreateStore(const_length, col_list_length_ptr);

  if (data_size != nullptr) {
    auto data_size_type = data_size->getType();
    llvm::Value* size_val = nullptr;
    if (data_size_type->isPointerTy()) {
      CHECK(data_size_type->getPointerElementType()->isIntegerTy(64));
      size_val =
          ir_builder.CreateLoad(data_size->getType()->getPointerElementType(), data_size);
    } else {
      CHECK(data_size_type->isIntegerTy(64));
      size_val = data_size;
    }
    ir_builder.CreateStore(size_val, col_list_size_ptr);
  } else {
    auto const_minus1 = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), -1, true);
    ir_builder.CreateStore(const_minus1, col_list_size_ptr);
  }

  auto col_list_ptr = ir_builder.CreatePointerCast(
      col_list_ptr_ptr,
      llvm::Type::getInt8PtrTy(ctx,
                               col_list_ptr_ptr->getType()->getPointerAddressSpace()));
  col_list_ptr->setName(col_list_name + "_ptrs");
  return col_list_ptr;
}

}  // namespace

std::shared_ptr<CompilationContext> TableFunctionCompilationContext::compile(
    const TableFunctionExecutionUnit& exe_unit,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);

  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  CHECK(cgen_state->module_ == nullptr);
  cgen_state->set_module_shallow_copy(
      executor_->getExtensionModuleContext()->getRTModule(/*is_l0=*/false));

  entry_point_func_ = generate_entry_point(cgen_state, co);

  generateEntryPoint(exe_unit,
                     /*is_gpu=*/co.device_type == ExecutorDeviceType::GPU,
                     co);

  if (co.device_type == ExecutorDeviceType::GPU) {
    generateGpuKernel(co);
  }
  return finalize(co);
}

bool TableFunctionCompilationContext::passColumnsByValue(
    const TableFunctionExecutionUnit& exe_unit,
    bool is_gpu) {
  auto mod = executor_->getExtModuleContext()->getRTUdfModule(is_gpu).get();
  if (mod != nullptr) {
    auto* flag = mod->getModuleFlag("pass_column_arguments_by_value");
    if (auto* cnt = llvm::mdconst::extract_or_null<llvm::ConstantInt>(flag)) {
      return cnt->getZExtValue();
    }
  }

  // fallback to original behavior
  return exe_unit.table_func.isRuntime();
}

void TableFunctionCompilationContext::generateTableFunctionCall(
    const TableFunctionExecutionUnit& exe_unit,
    const std::vector<llvm::Value*>& func_args,
    llvm::BasicBlock* bb_exit,
    llvm::Value* output_row_count_ptr) {
  auto cgen_state = executor_->getCgenStatePtr();
  // Emit llvm IR code to call the table function
  llvm::LLVMContext& ctx = cgen_state->context_;
  llvm::IRBuilder<>* ir_builder = &cgen_state->ir_builder_;

  std::string func_name = exe_unit.table_func.getName(false, true);
  llvm::Value* table_func_return =
      cgen_state->emitExternalCall(func_name, get_int_type(32, ctx), func_args);

  table_func_return->setName("table_func_ret");

  // If table_func_return is non-negative then store the value in
  // output_row_count and return zero. Otherwise, return
  // table_func_return that negative value contains the error code.
  llvm::BasicBlock* bb_exit_0 =
      llvm::BasicBlock::Create(ctx, ".exit0", entry_point_func_);

  llvm::Constant* const_zero =
      llvm::ConstantInt::get(table_func_return->getType(), 0, true);
  llvm::Value* is_ok = ir_builder->CreateICmpSGE(table_func_return, const_zero);
  ir_builder->CreateCondBr(is_ok, bb_exit_0, bb_exit);

  ir_builder->SetInsertPoint(bb_exit_0);
  llvm::Value* r =
      ir_builder->CreateIntCast(table_func_return, get_int_type(64, ctx), true);
  ir_builder->CreateStore(r, output_row_count_ptr);
  ir_builder->CreateRet(const_zero);

  ir_builder->SetInsertPoint(bb_exit);
  ir_builder->CreateRet(table_func_return);
}

void TableFunctionCompilationContext::generateEntryPoint(
    const TableFunctionExecutionUnit& exe_unit,
    bool is_gpu,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(entry_point_func_);
  CHECK_EQ(entry_point_func_->arg_size(), 5);
  auto arg_it = entry_point_func_->arg_begin();
  const auto mgr_ptr = &*arg_it;
  const auto input_cols_arg = &*(++arg_it);
  const auto input_row_counts_arg = &*(++arg_it);
  const auto output_buffers_arg = &*(++arg_it);
  const auto output_row_count_ptr = &*(++arg_it);
  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  llvm::BasicBlock* bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", entry_point_func_, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);

  llvm::BasicBlock* bb_exit = llvm::BasicBlock::Create(ctx, ".exit", entry_point_func_);

  llvm::BasicBlock* func_body_bb = llvm::BasicBlock::Create(
      ctx, ".func_body0", cgen_state->ir_builder_.GetInsertBlock()->getParent());

  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(func_body_bb);

  cgen_state->ir_builder_.SetInsertPoint(func_body_bb);
  auto col_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_cols_arg, cgen_state->ir_builder_, ctx);
  CHECK_EQ(exe_unit.input_exprs.size(), col_heads.size());
  auto row_count_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_row_counts_arg, cgen_state->ir_builder_, ctx);
  // The column arguments of C++ UDTFs processed by clang must be
  // passed by reference, see rbc issues 200 and 289.
  auto pass_column_by_value = passColumnsByValue(exe_unit, is_gpu);
  std::vector<llvm::Value*> func_args;
  size_t func_arg_index = 0;
  if (exe_unit.table_func.usesManager()) {
    func_args.push_back(mgr_ptr);
    func_arg_index++;
  }
  int col_index = -1;
  for (size_t i = 0; i < exe_unit.input_exprs.size(); i++) {
    const auto& expr = exe_unit.input_exprs[i];
    auto type = expr->type();
    if (col_index == -1) {
      func_arg_index += 1;
    }
    if (type->isFloatingPoint()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i],
          get_fp_ptr_type(get_bit_width(type),
                          ctx,
                          col_heads[i]->getType()->getPointerAddressSpace()));
      func_args.push_back(
          cgen_state->ir_builder_.CreateLoad(r->getType()->getPointerElementType(), r));
      CHECK_EQ(col_index, -1);
    } else if (type->isInteger() || type->isBoolean()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i],
          get_int_ptr_type(get_bit_width(type),
                           ctx,
                           col_heads[i]->getType()->getPointerAddressSpace()));
      func_args.push_back(
          cgen_state->ir_builder_.CreateLoad(r->getType()->getPointerElementType(), r));
      CHECK_EQ(col_index, -1);
    } else if (type->isString()) {
      auto varchar_size = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i],
          get_int_ptr_type(64, ctx, col_heads[i]->getType()->getPointerAddressSpace()));
      auto varchar_ptr = cgen_state->ir_builder_.CreateGEP(
          col_heads[i]->getType()->getScalarType()->getPointerElementType(),
          col_heads[i],
          cgen_state->llInt(8));
      auto [varchar_struct, varchar_struct_ptr] =
          alloc_column(std::string("varchar_literal.") + std::to_string(func_arg_index),
                       i,
                       type,
                       varchar_ptr,
                       varchar_size,
                       ctx,
                       cgen_state->ir_builder_,
                       co.codegen_traits_desc);
      func_args.push_back(
          (pass_column_by_value
               ? cgen_state->ir_builder_.CreateLoad(
                     varchar_struct->getType()->getPointerElementType(), varchar_struct)
               : varchar_struct_ptr));
      CHECK_EQ(col_index, -1);
    } else if (type->isColumn()) {
      auto [col, col_ptr] =
          alloc_column(std::string("input_col.") + std::to_string(func_arg_index),
                       i,
                       type->as<hdk::ir::ColumnType>()->columnType(),
                       col_heads[i],
                       row_count_heads[i],
                       ctx,
                       cgen_state->ir_builder_,
                       co.codegen_traits_desc);
      func_args.push_back((pass_column_by_value
                               ? cgen_state->ir_builder_.CreateLoad(
                                     col->getType()->getPointerElementType(), col)
                               : col_ptr));
      CHECK_EQ(col_index, -1);
    } else if (type->isColumnList()) {
      auto col_list_type = type->as<hdk::ir::ColumnListType>();
      if (col_index == -1) {
        auto col_list = alloc_column_list(
            std::string("input_col_list.") + std::to_string(func_arg_index),
            col_list_type->columnType(),
            col_heads[i],
            col_list_type->length(),
            row_count_heads[i],
            ctx,
            cgen_state->ir_builder_,
            co.codegen_traits_desc);
        func_args.push_back(col_list);
      }
      col_index++;
      if (col_index + 1 == col_list_type->length()) {
        col_index = -1;
      }
    } else {
      throw std::runtime_error(
          "Only integer and floating point columns or scalars are supported as inputs to "
          "table "
          "functions, got " +
          type->toString());
    }
  }
  std::vector<llvm::Value*> output_col_args;
  for (size_t i = 0; i < exe_unit.target_exprs.size(); i++) {
    auto* gep = cgen_state->ir_builder_.CreateGEP(
        output_buffers_arg->getType()->getScalarType()->getPointerElementType(),
        output_buffers_arg,
        cgen_state->llInt(i));
    auto output_load =
        cgen_state->ir_builder_.CreateLoad(gep->getType()->getPointerElementType(), gep);
    const auto& expr = exe_unit.target_exprs[i];
    auto type = expr->type();
    CHECK(!type->isColumn());      // UDTF output column type is its data type
    CHECK(!type->isColumnList());  // TODO: when UDTF outputs column_list, convert it to
                                   // output columns
    auto [col, col_ptr] = alloc_column(
        std::string("output_col.") + std::to_string(i),
        i,
        type,
        (is_gpu ? output_load : nullptr),  // CPU: set_output_row_size will set the output
                                           // Column ptr member
        output_row_count_ptr,
        ctx,
        cgen_state->ir_builder_,
        co.codegen_traits_desc);
    if (!is_gpu) {
      cgen_state->emitExternalCall(
          "TableFunctionManager_register_output_column",
          llvm::Type::getVoidTy(ctx),
          {mgr_ptr, llvm::ConstantInt::get(get_int_type(32, ctx), i, true), col_ptr});
    }
    output_col_args.push_back((pass_column_by_value ? col : col_ptr));
  }

  // output column members must be set before loading column when
  // column instances are passed by value
  if ((exe_unit.table_func.hasOutputSizeKnownPreLaunch()) && !is_gpu) {
    cgen_state->emitExternalCall(
        "TableFunctionManager_set_output_row_size",
        llvm::Type::getVoidTy(ctx),
        {mgr_ptr,
         cgen_state->ir_builder_.CreateLoad(
             output_row_count_ptr->getType()->getPointerElementType(),
             output_row_count_ptr)});
  }

  for (auto& col : output_col_args) {
    func_args.push_back((pass_column_by_value
                             ? cgen_state->ir_builder_.CreateLoad(
                                   col->getType()->getPointerElementType(), col)
                             : col));
  }

  generateTableFunctionCall(exe_unit, func_args, bb_exit, output_row_count_ptr);

  // std::cout << "=================================" << std::endl;
  // entry_point_func_->print(llvm::outs());
  // std::cout << "=================================" << std::endl;

  compiler::verify_function_ir(entry_point_func_);
}

void TableFunctionCompilationContext::generateGpuKernel(const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(entry_point_func_);
  compiler::CodegenTraits cgen_traits =
      compiler::CodegenTraits::get(co.codegen_traits_desc);
  std::vector<llvm::Type*> arg_types;
  arg_types.reserve(entry_point_func_->arg_size());
  std::for_each(entry_point_func_->arg_begin(),
                entry_point_func_->arg_end(),
                [&arg_types](const auto& arg) { arg_types.push_back(arg.getType()); });
  CHECK_EQ(arg_types.size(), entry_point_func_->arg_size());

  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  std::vector<llvm::Type*> wrapper_arg_types(arg_types.size() + 1);

  wrapper_arg_types[0] = cgen_traits.localPointerType(get_int_type(32, ctx));
  wrapper_arg_types[1] = arg_types[0];

  for (size_t i = 1; i < arg_types.size(); ++i) {
    wrapper_arg_types[i + 1] = arg_types[i];
  }

  auto wrapper_ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_arg_types, false);
  kernel_func_ = llvm::Function::Create(wrapper_ft,
                                        llvm::Function::ExternalLinkage,
                                        "table_func_kernel",
                                        cgen_state->module_);

  auto wrapper_bb_entry = llvm::BasicBlock::Create(ctx, ".entry", kernel_func_, 0);
  llvm::IRBuilder<> b(ctx);
  b.SetInsertPoint(wrapper_bb_entry);
  std::vector<llvm::Value*> loaded_args = {kernel_func_->arg_begin() + 1};
  for (size_t i = 2; i < wrapper_arg_types.size(); ++i) {
    loaded_args.push_back(kernel_func_->arg_begin() + i);
  }
  auto error_lv = b.CreateCall(entry_point_func_, loaded_args);
  b.CreateStore(error_lv, kernel_func_->arg_begin());
  b.CreateRetVoid();
}

std::shared_ptr<CompilationContext> TableFunctionCompilationContext::finalize(
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  /*
    TODO 1: eliminate need for OverrideFromSrc
    TODO 2: detect and link only the udf's that are needed
  */
  auto cgen_state = executor_->getCgenStatePtr();
  auto is_gpu = co.device_type == ExecutorDeviceType::GPU;
  if (executor_->has_rt_udf_module(is_gpu)) {
    CodeGenerator::link_udf_module(
        executor_->getExtModuleContext()->getRTUdfModule(is_gpu),
        *(cgen_state->module_),
        cgen_state,
        llvm::Linker::Flags::OverrideFromSrc);
  }

  // Add code to cache?

  LOG(IR) << "Table Function Entry Point IR\n"
          << serialize_llvm_object(entry_point_func_);
  GPUTarget target{};
  if (is_gpu) {
    LOG(IR) << "Table Function Kernel IR\n" << serialize_llvm_object(kernel_func_);
    CHECK(executor_);
    target = {executor_->gpuMgr(), executor_->blockSize(), cgen_state, false};
  }

  auto backend =
      compiler::getBackend(co.device_type,
                           executor_->getExtensionModuleContext()->getExtensionModules(),
                           /*is_gpu_smem_used=*/false,
                           target);
  auto ctx = backend->generateNativeCode(
      entry_point_func_, kernel_func_, {entry_point_func_, kernel_func_}, co);

  if (!is_gpu) {
    std::shared_ptr<CpuCompilationContext> cpu_ctx =
        std::dynamic_pointer_cast<CpuCompilationContext>(ctx);
    cpu_ctx->setFunctionPointer(entry_point_func_);
  }

  LOG(IR) << "End of IR";

  return ctx;
}
