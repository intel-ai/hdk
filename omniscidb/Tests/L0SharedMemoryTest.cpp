#include "GpuSharedMemoryTestHelpers.h"
#include "L0Mgr/L0Mgr.h"
#include "QueryEngine/Compiler/HelperFunctions.h"
#include "TestHelpers.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "LLVMSPIRVLib/LLVMSPIRVLib.h"

#include <iostream>

extern bool g_is_test_env;

auto int8_type = hdk::ir::Context::defaultCtx().int8();
auto int16_type = hdk::ir::Context::defaultCtx().int16();
auto int32_type = hdk::ir::Context::defaultCtx().int32();
auto int64_type = hdk::ir::Context::defaultCtx().int64();
auto float_type = hdk::ir::Context::defaultCtx().fp32();
auto double_type = hdk::ir::Context::defaultCtx().fp64();

class L0ReductionTester : public GpuReductionTester {
 public:
  L0ReductionTester(const Config& config,
                    llvm::Module* module,
                    llvm::LLVMContext& context,
                    const QueryMemoryDescriptor& qmd,
                    const std::vector<TargetInfo>& targets,
                    const std::vector<int64_t>& init_agg_values,
                    GpuMgr* gpu_mgr,
                    const compiler::CodegenTraits& traits,
                    Executor* executor)
      : GpuReductionTester(config,
                           module,
                           context,
                           qmd,
                           targets,
                           init_agg_values,
                           gpu_mgr,
                           traits,
                           executor) {}
  void performReductionTest(const std::vector<std::unique_ptr<ResultSet>>& result_sets,
                            const ResultSetStorage* gpu_result_storage,
                            const size_t device_id);
};

std::unique_ptr<L0DeviceCompilationContext> compile_and_link_l0_code(
    const std::string& l0_llir,
    llvm::Module* module,
    l0::L0Manager* l0_mgr,
    const std::string& kernel_name,
    const size_t gpu_block_size = 256,
    const size_t gpu_device_idx = 0) {
  CHECK(module);
  CHECK(l0_mgr);
  L0BinResult bin_result = spv_to_bin(l0_llir, kernel_name, gpu_block_size, l0_mgr);
  auto device_compilation_ctx = std::make_unique<L0DeviceCompilationContext>(
      bin_result.device, bin_result.kernel, bin_result.module, l0_mgr, 0, 0);
  return device_compilation_ctx;
}

void prepare_generated_l0_kernel(llvm::Module* module,
                                 llvm::LLVMContext& context,
                                 llvm::Function* kernel) {
  kernel->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  llvm::Metadata* spirv_src_ops[] = {
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 3 /*OpenCL_C*/)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(context), 102000 /*OpenCL ver 1.2*/))};
  llvm::NamedMDNode* spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(llvm::MDNode::get(context, spirv_src_ops));
}

void link_genx(llvm::Module* module, Executor* executor);

void L0ReductionTester::performReductionTest(
    const std::vector<std::unique_ptr<ResultSet>>& result_sets,
    const ResultSetStorage* gpu_result_storage,
    const size_t device_id) {
  link_genx(module_, executor_);
  prepare_generated_l0_kernel(module_, context_, getWrapperKernel());

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::ostringstream ss;
  std::string err;
  auto success = llvm::writeSpirv(module_, opts, ss, err);
  CHECK(success) << "Spirv translation failed with error: " << err << "\n";
  std::string module_str(ss.str());

  auto gpu_context = compile_and_link_l0_code(module_str,
                                              module_,
                                              dynamic_cast<l0::L0Manager*>(gpu_mgr_),
                                              getWrapperKernel()->getName().str());

  const auto buffer_size = query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU);
  const uint64_t num_buffers = static_cast<uint64_t>(result_sets.size());
  std::vector<int8_t*> d_input_buffers;
  for (size_t i = 0; i < num_buffers; i++) {
    d_input_buffers.push_back(gpu_mgr_->allocateDeviceMem(buffer_size, device_id));
    gpu_mgr_->copyHostToDevice(d_input_buffers[i],
                               result_sets[i]->getStorage()->getUnderlyingBuffer(),
                               buffer_size,
                               device_id);
  }

  constexpr size_t num_kernel_params = 3;
  CHECK_EQ(getWrapperKernel()->arg_size(), num_kernel_params);

  // parameter 1 : an array of device pointers
  std::vector<int8_t*> h_input_buffer_dptrs;
  h_input_buffer_dptrs.reserve(num_buffers);
  std::transform(d_input_buffers.begin(),
                 d_input_buffers.end(),
                 std::back_inserter(h_input_buffer_dptrs),
                 [](int8_t* dptr) { return dptr; });

  CHECK_EQ(h_input_buffer_dptrs.size(), (size_t)num_buffers);
  auto d_input_buffer_dptrs =
      gpu_mgr_->allocateDeviceMem(num_buffers * sizeof(int8_t*), device_id);
  gpu_mgr_->copyHostToDevice(d_input_buffer_dptrs,
                             reinterpret_cast<int8_t*>(h_input_buffer_dptrs.data()),
                             num_buffers * sizeof(int8_t*),
                             device_id);

  // parameter 3: device pointer to the output buffer
  auto d_result_buffer = gpu_mgr_->allocateDeviceMem(buffer_size, device_id);
  gpu_mgr_->copyHostToDevice(
      d_result_buffer, gpu_result_storage->getUnderlyingBuffer(), buffer_size, device_id);

  // launching a kernel:
  auto kernel = gpu_context->kernel();
  auto device = gpu_context->device();

  auto q = device->command_queue();
  auto q_list = device->create_command_list();

  // Leaving the kernel parameters empty and set them explicitly (the standard launch API
  // expects pointers only as kernel arguments)
  std::vector<int8_t*> h_kernel_params{};
  L0_SAFE_CALL(zeKernelSetArgumentValue(
      kernel->handle(), 0, sizeof(int8_t*), &d_input_buffer_dptrs));
  L0_SAFE_CALL(
      zeKernelSetArgumentValue(kernel->handle(), 1, sizeof(uint64_t), &num_buffers));
  L0_SAFE_CALL(
      zeKernelSetArgumentValue(kernel->handle(), 2, sizeof(int8_t*), &d_result_buffer));

  l0::GroupCount gc = {static_cast<uint32_t>(num_buffers), 1, 1};

  q_list->launch(kernel, h_kernel_params, gc);
  q_list->submit(*q.get());

  // transfer back the results:
  gpu_mgr_->copyDeviceToHost(
      gpu_result_storage->getUnderlyingBuffer(), d_result_buffer, buffer_size, device_id);

  // release the gpu memory used :
  for (auto& d_buffer : d_input_buffers) {
    gpu_mgr_->freeDeviceMem(d_buffer);
  }
  gpu_mgr_->freeDeviceMem(d_input_buffer_dptrs);
  gpu_mgr_->freeDeviceMem(d_result_buffer);
}

void link_genx(llvm::Module* module, Executor* executor) {
  DUMP_MODULE(module, "before.linking.spirv.ll")
  auto& ext_module = executor->getExtensionModuleContext()->getSpirvHelperFuncModule();
  compiler::insert_globals(ext_module.get(), module);
  for (auto& F : *(ext_module)) {
    compiler::insert_declaration(ext_module.get(), module, F.getName().str());
  }

  for (auto& F : *ext_module) {
    if (!F.isDeclaration()) {
      compiler::replace_function(ext_module.get(), module, F.getName().str());
    }
  }

  // set proper calling conv & mangle spirv built-ins
  for (auto& Fn : *module) {
    Fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
    if (Fn.getName().startswith("__spirv_")) {
      CHECK(Fn.isDeclaration());
      Fn.setName(compiler::mangle_spirv_builtin(Fn));
    }
  }

  for (auto& Fn : *module) {
    for (auto I = llvm::inst_begin(Fn), E = llvm::inst_end(Fn); I != E; ++I) {
      if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
        CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
      }
    }
  }
  DUMP_MODULE(module, "after.linking.spirv.ll")
}

void perform_test_and_verify_results(TestInputData input) {
  const auto platform = GpuMgrPlatform::L0;
  const bool is_l0 = platform == GpuMgrPlatform::L0;
  auto executor = Executor::getExecutor(nullptr, nullptr);
  auto& context = executor->getContext();
  auto cgen_state = std::unique_ptr<CgenState>(
      new CgenState({}, false, false, executor->getExtensionModuleContext(), context));
  cgen_state->set_module_shallow_copy(
      executor->getExtensionModuleContext()->getRTModule(is_l0));
  auto module = cgen_state->module_;
  link_genx(module, executor.get());
  auto cgen_traits = get_codegen_traits(platform);
  module->setDataLayout(cgen_traits.dataLayout());
  module->setTargetTriple(cgen_traits.triple());
  auto l0_mgr = std::make_unique<l0::L0Manager>();
  const auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(nullptr, Executor::getArenaBlockSize());
  auto query_mem_desc = perfect_hash_one_col_desc(
      input.target_infos, input.suggested_agg_widths, input.min_entry, input.max_entry);
  if (input.keyless_hash) {
    query_mem_desc.setHasKeylessHash(true);
    query_mem_desc.setTargetIdxForKey(input.target_index_for_key);
  }

  std::vector<StrideNumberGenerator> generators(
      input.num_input_buffers, StrideNumberGenerator(1, input.step_size));
  std::vector<size_t> steps(input.num_input_buffers, input.step_size);
  auto input_result_sets = create_and_fill_input_result_sets(input.num_input_buffers,
                                                             row_set_mem_owner,
                                                             query_mem_desc,
                                                             input.target_infos,
                                                             generators,
                                                             steps);

  const auto [cpu_result_set, gpu_result_set] = create_and_init_output_result_sets(
      row_set_mem_owner, query_mem_desc, input.target_infos);

  // performing reduciton using the GPU reduction code:
  Config config;
  L0ReductionTester gpu_smem_tester(config,
                                    module,
                                    context,
                                    query_mem_desc,
                                    input.target_infos,
                                    init_agg_val_vec(input.target_infos, query_mem_desc),
                                    l0_mgr.get(),
                                    cgen_traits,
                                    executor.get());
  gpu_smem_tester.codegen(CompilationOptions::defaults(
      ExecutorDeviceType::GPU,
      is_l0));  // generate code for gpu reduciton and initialization
  gpu_smem_tester.codegenWrapperKernel();
  gpu_smem_tester.performReductionTest(
      input_result_sets, gpu_result_set->getStorage(), input.device_id);

  // CPU reduction for validation:
  perform_reduction_on_cpu(input_result_sets, cpu_result_set->getStorage());

  const auto cmp_result =
      std::memcmp(cpu_result_set->getStorage()->getUnderlyingBuffer(),
                  gpu_result_set->getStorage()->getUnderlyingBuffer(),
                  query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU));
  ASSERT_EQ(cmp_result, 0);
}

TEST(Smoke, Count) {
  std::vector<const hdk::ir::Type*> variants = {
      int32_type, int64_type, float_type, double_type};
  for (auto* type : variants) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(1)
        .setTargetInfos(generate_custom_agg_target_infos(
            {1}, {hdk::ir::AggType::kCount}, {type}, {type}))
        .setAggWidth(type->size())
        .setMinEntry(0)
        .setMaxEntry(10)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(Smoke, Min) {
  std::vector<const hdk::ir::Type*> variants = {
      int32_type, int64_type, float_type, double_type};
  for (auto* type : variants) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {1}, {hdk::ir::AggType::kMin}, {type}, {type}))
        .setAggWidth(type->size())
        .setMinEntry(0)
        .setMaxEntry(10)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(Smoke, Max) {
  std::vector<const hdk::ir::Type*> variants = {
      int32_type, int64_type, float_type, double_type};
  for (auto* type : variants) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {1}, {hdk::ir::AggType::kMax}, {type}, {type}))
        .setAggWidth(type->size())
        .setMinEntry(0)
        .setMaxEntry(10)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(Smoke, Sum) {
  std::vector<const hdk::ir::Type*> variants = {
      int32_type, int64_type, float_type, double_type};
  for (auto* type : variants) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {1}, {hdk::ir::AggType::kSum}, {type}, {type}))
        .setAggWidth(type->size())
        .setMinEntry(0)
        .setMaxEntry(10)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(Smoke, Avg) {
  std::vector<const hdk::ir::Type*> variants = {
      int32_type, int64_type, float_type, double_type};
  for (auto* type : variants) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {1}, {hdk::ir::AggType::kAvg}, {type}, {type}))
        .setAggWidth(8)  // type->size()?
        .setMinEntry(0)
        .setMaxEntry(10)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableEntries_CountQuery_4B_Group) {
  // Gen9 has a max group size of 256
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241 /*, 511, 1021*/}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {4}, {hdk::ir::AggType::kCount}, {int32_type}, {int32_type}))
        .setAggWidth(4)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableEntries_CountQuery_8B_Group) {
  // Gen9 has a max group size of 256
  for (auto num_entries : {1, 2, 3, 5, 13, 31, 63, 126, 241 /*, 511, 1021*/}) {
    TestInputData input;
    input.setDeviceId(0)
        .setNumInputBuffers(4)
        .setTargetInfos(generate_custom_agg_target_infos(
            {8}, {hdk::ir::AggType::kCount}, {int64_type}, {int64_type}))
        .setAggWidth(8)
        .setMinEntry(0)
        .setMaxEntry(num_entries)
        .setStepSize(2)
        .setKeylessHash(true)
        .setTargetIndexForKey(0);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_1) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kMin,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kAvg},
          {int64_type, int64_type, int64_type, int64_type, double_type},
          {int32_type, int32_type, int32_type, int32_type, int32_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_2) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(126)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMin},
          {int64_type, double_type, int64_type, int64_type, int64_type},
          {int32_type, int32_type, int32_type, int32_type, int32_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_3) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(204)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMin},
          {int64_type, double_type, double_type, double_type, double_type},
          {int32_type, double_type, double_type, double_type, double_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableSteps_FixedEntries_4) {
  TestInputData input;
  input.setDeviceId(0)
      .setNumInputBuffers(4)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(255)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMin},
          {int64_type, float_type, float_type, float_type, float_type},
          {int16_type, float_type, float_type, float_type, float_type}));

  for (auto& step_size : {2, 3, 5, 7, 11, 13}) {
    input.setStepSize(step_size);
    perform_test_and_verify_results(input);
  }
}

TEST(SingleColumn, VariableNumBuffers) {
  TestInputData input;
  input.setDeviceId(0)
      .setAggWidth(8)
      .setMinEntry(0)
      .setMaxEntry(255)
      .setKeylessHash(true)
      .setTargetIndexForKey(0)
      .setTargetInfos(generate_custom_agg_target_infos(
          {8},
          {hdk::ir::AggType::kCount,
           hdk::ir::AggType::kSum,
           hdk::ir::AggType::kAvg,
           hdk::ir::AggType::kMax,
           hdk::ir::AggType::kMin},
          {int32_type, int64_type, double_type, float_type, double_type},
          {int8_type, int8_type, int16_type, float_type, double_type}));

  for (auto& num_buffers : {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128}) {
    input.setNumInputBuffers(num_buffers);
    perform_test_and_verify_results(input);
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
