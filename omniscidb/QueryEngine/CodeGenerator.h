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

#pragma once

#include <llvm/IR/Value.h>

#include "IR/Expr.h"

#include "Compiler/CodegenTraitsDescriptor.h"
#include "Execute.h"
#include "QueryEngine/Target.h"

// Code generation utility to be used for queries and scalar expressions.
class CodeGenerator {
 public:
  CodeGenerator(Executor* executor, compiler::CodegenTraitsDescriptor cgen_traits_desc)
      : executor_(executor)
      , config_(executor->getConfig())
      , cgen_state_(executor->cgen_state_.get())
      , plan_state_(executor->plan_state_.get())
      , codegen_traits_desc(cgen_traits_desc) {}

  // Overload which can be used without an executor, for SQL scalar expression code
  // generation.
  CodeGenerator(const Config& config,
                CgenState* cgen_state,
                PlanState* plan_state,
                compiler::CodegenTraitsDescriptor cgen_traits_desc)
      : executor_(nullptr)
      , config_(config)
      , cgen_state_(cgen_state)
      , plan_state_(plan_state)
      , codegen_traits_desc(cgen_traits_desc) {}

  // Generates IR value(s) for the given analyzer expression.
  std::vector<llvm::Value*> codegen(const hdk::ir::Expr*,
                                    const bool fetch_columns,
                                    const CompilationOptions&);

  // Generates constant values in the literal buffer of a query.
  std::vector<llvm::Value*> codegenHoistedConstants(
      const std::vector<const hdk::ir::Constant*>& constants,
      bool use_dict_encoding,
      int dict_id);

  static llvm::ConstantInt* codegenIntConst(const hdk::ir::Constant* constant,
                                            CgenState* cgen_state);

  llvm::Value* codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                          const hdk::ir::Type* operand_type,
                                          const hdk::ir::Type* type,
                                          bool upscale = true);

  void codegenCastBetweenIntTypesOverflowChecks(llvm::Value* operand_lv,
                                                const hdk::ir::Type* operand_type,
                                                const hdk::ir::Type* type,
                                                const int64_t scale);

  // Generates the index of the current row in the context of query execution.
  llvm::Value* posArg(const hdk::ir::Expr*) const;

  llvm::Value* toBool(llvm::Value*);

  llvm::Value* castArrayPointer(llvm::Value* ptr, const hdk::ir::Type* elem_type);

  static std::unordered_set<llvm::Function*> markDeadRuntimeFuncs(
      llvm::Module& module,
      const std::vector<llvm::Function*>& roots,
      const std::vector<llvm::Function*>& leaves);

  static bool alwaysCloneRuntimeFunction(const llvm::Function* func);

  static void link_udf_module(const std::unique_ptr<llvm::Module>& udf_module,
                              llvm::Module& module,
                              CgenState* cgen_state,
                              llvm::Linker::Flags flags = llvm::Linker::Flags::None);

  static bool prioritizeQuals(const RelAlgExecutionUnit& ra_exe_unit,
                              std::vector<const hdk::ir::Expr*>& primary_quals,
                              std::vector<const hdk::ir::Expr*>& deferred_quals,
                              const PlanState::HoistedFiltersSet& hoisted_quals);

  struct ExecutorRequired : public std::runtime_error {
    ExecutorRequired()
        : std::runtime_error("Executor required to generate this expression") {}
  };

  struct NullCheckCodegen {
    NullCheckCodegen(CgenState* cgen_state,
                     Executor* executor,
                     llvm::Value* nullable_lv,
                     const hdk::ir::Type* nullable_type,
                     const std::string& name = "");

    llvm::Value* finalize(llvm::Value* null_lv, llvm::Value* notnull_lv);

    CgenState* cgen_state{nullptr};
    std::string name;
    llvm::BasicBlock* nullcheck_bb{nullptr};
    llvm::PHINode* nullcheck_value{nullptr};
    std::unique_ptr<DiamondCodegen> null_check;
  };

 private:
  std::vector<llvm::Value*> codegen(const hdk::ir::Constant*,
                                    bool use_dict_encoding,
                                    int dict_id,
                                    const CompilationOptions&);

  virtual std::vector<llvm::Value*> codegenColumn(const hdk::ir::ColumnVar*,
                                                  const bool fetch_column,
                                                  const CompilationOptions&);

  llvm::Value* codegenArith(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegenUMinus(const hdk::ir::UOper*, const CompilationOptions&);

  llvm::Value* codegenCmp(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegenCmp(hdk::ir::OpType,
                          hdk::ir::Qualifier,
                          std::vector<llvm::Value*>,
                          const hdk::ir::Type*,
                          const hdk::ir::Expr*,
                          const CompilationOptions&);

  llvm::Value* codegenIsNull(const hdk::ir::UOper*, const CompilationOptions&);

  llvm::Value* codegenIsNullNumber(llvm::Value*, const hdk::ir::Type*);

  llvm::Value* codegenLogical(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegenLogical(const hdk::ir::UOper*, const CompilationOptions&);

  llvm::Value* codegenCast(const hdk::ir::UOper*, const CompilationOptions&);

  llvm::Value* codegenCast(llvm::Value* operand_lv,
                           const hdk::ir::Type* operand_type,
                           const hdk::ir::Type* type,
                           const bool operand_is_const,
                           bool is_dict_intersection,
                           const CompilationOptions& co);

  llvm::Value* codegen(const hdk::ir::InValues*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::InIntegerSet* expr, const CompilationOptions& co);

  std::vector<llvm::Value*> codegen(const hdk::ir::CaseExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::ExtractExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::DateAddExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::DateDiffExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::DateTruncExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::CharLengthExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::KeyForStringExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::SampleRatioExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::WidthBucketExpr*, const CompilationOptions&);

  llvm::Value* codegenConstantWidthBucketExpr(const hdk::ir::WidthBucketExpr*,
                                              bool,
                                              const CompilationOptions&);

  llvm::Value* codegenWidthBucketExpr(const hdk::ir::WidthBucketExpr*,
                                      bool,
                                      const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::LowerExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::LikeExpr*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::RegexpExpr*, const CompilationOptions&);

  llvm::Value* codegenUnnest(const hdk::ir::UOper*, const CompilationOptions&);

  llvm::Value* codegenArrayAt(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::CardinalityExpr*, const CompilationOptions&);

  std::vector<llvm::Value*> codegenArrayExpr(const hdk::ir::ArrayExpr*,
                                             const CompilationOptions&);

  llvm::Value* codegenFunctionOper(const hdk::ir::FunctionOper*,
                                   const CompilationOptions&);

  llvm::Value* codegenFunctionOperWithCustomTypeHandling(
      const hdk::ir::FunctionOperWithCustomTypeHandling*,
      const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegen(const hdk::ir::UOper*, const CompilationOptions&);

  std::vector<llvm::Value*> codegenHoistedConstantsLoads(const hdk::ir::Type* type,
                                                         const bool use_dict_encoding,
                                                         const int dict_id,
                                                         const int16_t lit_off);

  std::vector<llvm::Value*> codegenHoistedConstantsPlaceholders(
      const hdk::ir::Type* type,
      const bool use_dict_encoding,
      const int16_t lit_off,
      const std::vector<llvm::Value*>& literal_loads);

  std::vector<llvm::Value*> codegenColVar(const hdk::ir::ColumnVar*,
                                          const bool fetch_column,
                                          const bool update_query_plan,
                                          const CompilationOptions&);

  llvm::Value* codegenFixedLengthColVar(const hdk::ir::ColumnVar* col_var,
                                        llvm::Value* col_byte_stream,
                                        llvm::Value* pos_arg);

  // Generates code for a fixed length column when a window function is active.
  llvm::Value* codegenFixedLengthColVarInWindow(const hdk::ir::ColumnVar* col_var,
                                                llvm::Value* col_byte_stream,
                                                llvm::Value* pos_arg);

  // Generate the position for the given window function and the query iteration position.
  llvm::Value* codegenWindowPosition(WindowFunctionContext* window_func_context,
                                     llvm::Value* pos_arg);

  std::vector<llvm::Value*> codegenVariableLengthStringColVar(
      llvm::Value* col_byte_stream,
      llvm::Value* pos_arg);

  llvm::Value* codegenRowId(const hdk::ir::ColumnVar* col_var,
                            const CompilationOptions& co);

  llvm::Value* codgenAdjustFixedEncNull(llvm::Value*, const hdk::ir::Type*);

  std::vector<llvm::Value*> codegenOuterJoinNullPlaceholder(
      const hdk::ir::ColumnVar* col_var,
      const bool fetch_column,
      const CompilationOptions& co);

  llvm::Value* codegenIntArith(const hdk::ir::BinOper*,
                               llvm::Value*,
                               llvm::Value*,
                               const CompilationOptions&);

  llvm::Value* codegenFpArith(const hdk::ir::BinOper*, llvm::Value*, llvm::Value*);

  llvm::Value* codegenCastTimestampToDate(llvm::Value* ts_lv,
                                          const hdk::ir::TimeUnit unit,
                                          const bool nullable);

  llvm::Value* codegenCastTimestampToTime(llvm::Value* ts_lv,
                                          const hdk::ir::Type* operand_type,
                                          const hdk::ir::Type* target_type);

  llvm::Value* codegenCastBetweenTimestamps(llvm::Value* ts_lv,
                                            const hdk::ir::Type* operand_type,
                                            const hdk::ir::Type* target_type);

  llvm::Value* codegenCastFromString(llvm::Value* operand_lv,
                                     const hdk::ir::Type* operand_type,
                                     const hdk::ir::Type* type,
                                     const bool operand_is_const,
                                     bool is_dict_intersection,
                                     const CompilationOptions& co);

  llvm::Value* codegenCastToFp(llvm::Value* operand_lv,
                               const hdk::ir::Type* operand_type,
                               const hdk::ir::Type* type);

  llvm::Value* codegenCastFromFp(llvm::Value* operand_lv,
                                 const hdk::ir::Type* operand_type,
                                 const hdk::ir::Type* type);

  llvm::Value* codegenAdd(const hdk::ir::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const hdk::ir::Type*,
                          const CompilationOptions&);

  llvm::Value* codegenSub(const hdk::ir::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const hdk::ir::Type*,
                          const CompilationOptions&);

  void codegenSkipOverflowCheckForNull(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       llvm::BasicBlock* no_overflow_bb,
                                       const hdk::ir::Type* type);

  llvm::Value* codegenMul(const hdk::ir::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const hdk::ir::Type*,
                          const CompilationOptions&,
                          bool downscale = true);

  llvm::Value* codegenDiv(llvm::Value* lhs_lv,
                          llvm::Value* rhs_lv,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const hdk::ir::Type* type,
                          bool upscale = true);

  llvm::Value* codegenDeciDiv(const hdk::ir::BinOper*, const CompilationOptions&);

  llvm::Value* codegenMod(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const hdk::ir::Type*);

  llvm::Value* codegenCase(const hdk::ir::CaseExpr*,
                           llvm::Type* case_llvm_type,
                           const bool is_real_str,
                           const CompilationOptions&);

  llvm::Value* codegenExtractHighPrecisionTimestamps(llvm::Value*,
                                                     const hdk::ir::Type*,
                                                     const hdk::ir::DateExtractField&);

  llvm::Value* codegenDateTruncHighPrecisionTimestamps(llvm::Value*,
                                                       const hdk::ir::Type*,
                                                       const hdk::ir::DateTruncField&);

  llvm::Value* codegenCmpDecimalConst(hdk::ir::OpType,
                                      hdk::ir::Qualifier,
                                      const hdk::ir::Expr*,
                                      const hdk::ir::Type*,
                                      const hdk::ir::Expr*,
                                      const CompilationOptions&);

  llvm::Value* codegenStrCmp(hdk::ir::OpType,
                             hdk::ir::Qualifier,
                             const hdk::ir::ExprPtr,
                             const hdk::ir::ExprPtr,
                             const CompilationOptions&);

  llvm::Value* codegenQualifierCmp(hdk::ir::OpType,
                                   hdk::ir::Qualifier,
                                   std::vector<llvm::Value*>,
                                   const hdk::ir::Expr*,
                                   const CompilationOptions&);

  llvm::Value* codegenLogicalShortCircuit(const hdk::ir::BinOper*,
                                          const CompilationOptions&);

  llvm::Value* codegenDictLike(const hdk::ir::ExprPtr arg,
                               const hdk::ir::Constant* pattern,
                               const bool ilike,
                               const bool is_simple,
                               const char escape_char,
                               const CompilationOptions&);

  llvm::Value* codegenDictStrCmp(const hdk::ir::ExprPtr,
                                 const hdk::ir::ExprPtr,
                                 hdk::ir::OpType,
                                 const CompilationOptions& co);

  llvm::Value* codegenDictRegexp(const hdk::ir::ExprPtr arg,
                                 const hdk::ir::Constant* pattern,
                                 const char escape_char,
                                 const CompilationOptions&);

  // Returns the IR value which holds true iff at least one match has been found for outer
  // join, null if there's no outer join condition on the given nesting level.
  llvm::Value* foundOuterJoinMatch(const size_t nesting_level) const;

  llvm::Value* resolveGroupedColumnReference(const hdk::ir::ColumnVar*);

  llvm::Value* colByteStream(const hdk::ir::ColumnVar* col_var,
                             const bool fetch_column,
                             const bool hoist_literals);

  hdk::ir::ExprPtr hashJoinLhs(const hdk::ir::ColumnVar* rhs) const;

  std::shared_ptr<const hdk::ir::ColumnVar> hashJoinLhsTuple(
      const hdk::ir::ColumnVar* rhs,
      const hdk::ir::BinOper* tautological_eq) const;

  std::unique_ptr<InValuesBitmap> createInValuesBitmap(const hdk::ir::InValues*,
                                                       const CompilationOptions&);

  bool checkExpressionRanges(const hdk::ir::UOper*, int64_t, int64_t);

  bool checkExpressionRanges(const hdk::ir::BinOper*, int64_t, int64_t);

  struct ArgNullcheckBBs {
    llvm::BasicBlock* args_null_bb;
    llvm::BasicBlock* args_notnull_bb;
    llvm::BasicBlock* orig_bb;
  };

  std::tuple<ArgNullcheckBBs, llvm::Value*> beginArgsNullcheck(
      const hdk::ir::FunctionOper* function_oper,
      const std::vector<llvm::Value*>& orig_arg_lvs);

  llvm::Value* endArgsNullcheck(const ArgNullcheckBBs&,
                                llvm::Value*,
                                llvm::Value*,
                                const hdk::ir::FunctionOper*,
                                const CompilationOptions& co);

  llvm::Value* codegenFunctionOperNullArg(const hdk::ir::FunctionOper*,
                                          const std::vector<llvm::Value*>&);

  void codegenBufferArgs(const std::string& udf_func_name,
                         size_t param_num,
                         llvm::Value* buffer_buf,
                         llvm::Value* buffer_size,
                         llvm::Value* buffer_is_null,
                         std::vector<llvm::Value*>& output_args);

  std::vector<llvm::Value*> codegenFunctionOperCastArgs(
      const hdk::ir::FunctionOper*,
      const ExtensionFunction*,
      const std::vector<llvm::Value*>&,
      const std::vector<size_t>&,
      const std::unordered_map<llvm::Value*, llvm::Value*>&,
      const CompilationOptions&);

  // Return LLVM intrinsic providing fast arithmetic with overflow check
  // for the given binary operation.
  llvm::Function* getArithWithOverflowIntrinsic(const hdk::ir::BinOper* bin_oper,
                                                llvm::Type* type);

  // Generate code for the given binary operation with overflow check.
  // Signed integer add, sub and mul operations are supported. Overflow
  // check is performed using LLVM arithmetic intrinsics which are not
  // supported for GPU. Return the IR value which holds operation result.
  llvm::Value* codegenBinOpWithOverflowForCPU(const hdk::ir::BinOper* bin_oper,
                                              llvm::Value* lhs_lv,
                                              llvm::Value* rhs_lv,
                                              const std::string& null_check_suffix,
                                              const hdk::ir::Type* type);

  Executor* executor_;

 protected:
  Executor* executor() const {
    if (!executor_) {
      throw ExecutorRequired();
    }
    return executor_;
  }

  const Config& config_;
  CgenState* cgen_state_;
  PlanState* plan_state_;
  compiler::CodegenTraitsDescriptor codegen_traits_desc;
  friend class RowFuncBuilder;
};

// Code generator specialized for scalar expressions which doesn't require an executor.
class ScalarCodeGenerator : public CodeGenerator {
 public:
  // Constructor which takes the runtime module.
  ScalarCodeGenerator(const Config& config,
                      std::unique_ptr<llvm::Module> llvm_module,
                      compiler::CodegenTraitsDescriptor cgen_desc)
      : CodeGenerator(config, nullptr, nullptr, cgen_desc)
      , module_(std::move(llvm_module)) {}

  // Function generated for a given analyzer expression. For GPU, a wrapper which meets
  // the kernel signature constraints (returns void, takes all arguments as pointers) is
  // generated. Also returns the list of column expressions for which compatible input
  // parameters must be passed to the input of the generated function.
  struct CompiledExpression {
    llvm::Function* func;
    llvm::Function* wrapper_func;
    std::vector<std::shared_ptr<const hdk::ir::ColumnVar>> inputs;
  };

  // Compiles the given scalar expression to IR and the list of columns in the expression,
  // needed to provide inputs to the generated function.
  CompiledExpression compile(const hdk::ir::Expr* expr,
                             const bool fetch_columns,
                             const CompilationOptions& co,
                             const compiler::CodegenTraits& traits);

  // Generates the native function pointers for each device.
  // NB: this is separated from the compile method to allow building higher level code
  // generators which can inline the IR for evaluating a single expression (for example
  // loops).
  std::vector<void*> generateNativeCode(Executor* executor,
                                        const CompiledExpression& compiled_expression,
                                        const CompilationOptions& co);

  GpuMgr* getGpuMgr() const { return gpu_mgr_.get(); }

  using ColumnMap =
      std::unordered_map<InputColDescriptor, std::shared_ptr<const hdk::ir::ColumnVar>>;

 private:
  std::vector<llvm::Value*> codegenColumn(const hdk::ir::ColumnVar*,
                                          const bool fetch_column,
                                          const CompilationOptions&) override;

  // Collect the columns used by the given analyzer expressions and fills in the column
  // map to be used during code generation.
  ColumnMap prepare(const hdk::ir::Expr*);

  std::vector<void*> generateNativeGPUCode(Executor* executor,
                                           llvm::Function* func,
                                           llvm::Function* wrapper_func,
                                           const CompilationOptions& co);

  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<CgenState> own_cgen_state_;
  std::unique_ptr<PlanState> own_plan_state_;
  std::unique_ptr<GpuMgr> gpu_mgr_;
  std::shared_ptr<CompilationContext> gpu_compilation_context_;
  std::shared_ptr<CpuCompilationContext> cpu_compilation_context_;
  std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;
};

/**
 * Makes a shallow copy (just declarations) of the runtime module. Function definitions
 * are cloned only if they're used from the generated code.
 */
std::unique_ptr<llvm::Module> runtime_module_shallow_copy(CgenState* cgen_state);

/**
 *  Loads individual columns from a single, packed pointers buffer (the byte stream arg)
 */
std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Value* byte_stream_arg,
                                                     llvm::IRBuilder<>& ir_builder,
                                                     llvm::LLVMContext& ctx);
