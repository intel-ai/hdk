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

#include "Shared/funcannotations.h"
#define QUERYENGINE_EXPORT RUNTIME_EXPORT

#include "CardinalityEstimator.h"
#include "QueryEngine/Execute.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <tbb/parallel_for.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <tbb/parallel_reduce.h>
#include <chrono>
#include <ctime>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "DataProvider/DictDescriptor.h"
#include "OSDependent/omnisci_path.h"
#include "QueryEngine/AggregateUtils.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/AnalyticalTemplatesExtractor.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/CostModel/Dispatchers/DefaultExecutionPolicy.h"
#include "QueryEngine/CostModel/Dispatchers/ProportionBasedExecutionPolicy.h"
#include "QueryEngine/CostModel/Dispatchers/RRExecutionPolicy.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/DynamicWatchdog.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExecutionKernel.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/InPlaceSort.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryRewrite.h"
#include "QueryEngine/QueryTemplateGenerator.h"
#include "QueryEngine/ResultSetReduction.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/SpeculativeTopN.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/UnnestedVarsCollector.h"
#include "QueryEngine/Visitors/TransientStringLiteralsVisitor.h"
#include "ResultSet/ColRangeInfo.h"
#include "Shared/checked_alloc.h"
#include "Shared/funcannotations.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "ThirdParty/robin_hood.h"

#include "CostModel/IterativeCostModel.h"

using namespace std::string_literals;

extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;
EXTERN extern bool g_is_test_env;

int const Executor::max_gpu_count;

const int32_t Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES;

std::map<ExtModuleKinds, std::string> Executor::extension_module_sources;

extern std::unique_ptr<llvm::Module> read_llvm_module_from_bc_file(
    const std::string& udf_ir_filename,
    llvm::LLVMContext& ctx);
extern std::unique_ptr<llvm::Module> read_llvm_module_from_ir_file(
    const std::string& udf_ir_filename,
    llvm::LLVMContext& ctx,
    bool is_gpu = false);
extern std::unique_ptr<llvm::Module> read_llvm_module_from_ir_string(
    const std::string& udf_ir_string,
    llvm::LLVMContext& ctx,
    bool is_gpu = false);

std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> Executor::s_stubs_accessor;
std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> Executor::s_code_accessor;
std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> Executor::cpu_code_accessor;
std::unique_ptr<CodeCacheAccessor<CompilationContext>> Executor::gpu_code_accessor;
size_t Executor::code_cache_size;
namespace {

void init_code_caches() {
  Executor::s_stubs_accessor = std::make_unique<CodeCacheAccessor<CpuCompilationContext>>(
      Executor::code_cache_size, "s_stubs_cache");
  Executor::s_code_accessor = std::make_unique<CodeCacheAccessor<CpuCompilationContext>>(
      Executor::code_cache_size, "s_code_cache");
  Executor::cpu_code_accessor =
      std::make_unique<CodeCacheAccessor<CpuCompilationContext>>(
          Executor::code_cache_size, "cpu_code_cache");
  Executor::gpu_code_accessor = std::make_unique<CodeCacheAccessor<CompilationContext>>(
      Executor::code_cache_size, "gpu_code_cache");
}

}  // namespace

/**
 * Flushes and re-initializes the code caches. Any cached references will be dropped. The
 * re-initialized code caches will be empty, allowing for higher-level buffer mgrs to be
 * torn down. If code caches are used, this must be called before the DataMgr global is
 * destroyed at exit.
 */
void Executor::resetCodeCache() {
  s_stubs_accessor.reset();
  s_code_accessor.reset();
  cpu_code_accessor.reset();
  gpu_code_accessor.reset();
  init_code_caches();
}

Executor::Executor(const ExecutorId executor_id,
                   Data_Namespace::DataMgr* data_mgr,
                   ConfigPtr config,
                   const std::string& debug_dir,
                   const std::string& debug_file)
    : executor_id_(executor_id)
    , context_(new llvm::LLVMContext())
    , config_(config)
    , block_size_x_(config->exec.override_gpu_block_size)
    , grid_size_x_(config->exec.override_gpu_grid_size)
    , debug_dir_(debug_dir)
    , debug_file_(debug_file)
    , data_mgr_(data_mgr)
    , temporary_tables_(nullptr)
    , input_table_info_cache_(this)
    , thread_id_(logger::thread_id()) {
  if (executor_id_ > INVALID_EXECUTOR_ID - 1) {
    throw std::runtime_error("Too many executors!");
  }

  extension_module_context_ = std::make_unique<ExtensionModuleContext>();
  cgen_state_ = std::make_unique<CgenState>(
      0, false, false, extension_module_context_.get(), getContext());

  std::call_once(first_init_flag_, [this]() {
    query_plan_dag_cache_ =
        std::make_unique<QueryPlanDagCache>(config_->cache.dag_cache_size);
    code_cache_size = config_->cache.code_cache_size;
    init_code_caches();
  });
  Executor::initialize_extension_module_sources();
  update_extension_modules();

  if (config_->exec.enable_cost_model) {
    try {
      cost_model = std::make_shared<costmodel::IterativeCostModel>();
      cost_model->calibrate({{ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}});
    } catch (costmodel::CostModelException& e) {
      LOG(DEBUG1) << "Cost model will be disabled due to creation error: " << e.what();
    }
  }
}

std::shared_ptr<costmodel::CostModel> Executor::getCostModel() {
  return cost_model;
}

void Executor::initialize_extension_module_sources() {
  if (Executor::extension_module_sources.find(ExtModuleKinds::template_module) ==
      Executor::extension_module_sources.end()) {
    auto root_path = omnisci::get_root_abs_path();
    auto template_path = root_path + "/QueryEngine/RuntimeFunctions.bc";
    CHECK(boost::filesystem::exists(template_path)) << template_path;
    Executor::extension_module_sources[ExtModuleKinds::template_module] = template_path;
#ifdef HAVE_CUDA
    auto rt_libdevice_path = get_cuda_home() + "/nvvm/libdevice/libdevice.10.bc";
    if (boost::filesystem::exists(rt_libdevice_path)) {
      Executor::extension_module_sources[ExtModuleKinds::rt_libdevice_module] =
          rt_libdevice_path;
    } else {
      LOG(WARNING) << "File " << rt_libdevice_path
                   << " does not exist; support for some UDF "
                      "functions might not be available.";
    }
#endif
#ifdef HAVE_L0
    auto l0_template_path = root_path + "/QueryEngine/RuntimeFunctionsL0.bc";
    CHECK(boost::filesystem::exists(l0_template_path));
    Executor::extension_module_sources[ExtModuleKinds::l0_template_module] =
        l0_template_path;
    auto genx_path = root_path + "/QueryEngine/genx.bc";
    CHECK(boost::filesystem::exists(genx_path));
    Executor::extension_module_sources[ExtModuleKinds::spirv_helper_funcs_module] =
        genx_path;
#endif
  }
}

void Executor::reset(const bool discard_runtime_modules_only) {
  // TODO: keep cached results that do not depend on runtime UDF
  s_code_accessor->clear();
  s_stubs_accessor->clear();
  cpu_code_accessor->clear();
  gpu_code_accessor->clear();

  CHECK(extension_module_context_);
  extension_module_context_->clear(discard_runtime_modules_only);

  if (discard_runtime_modules_only) {
    cgen_state_->module_ = nullptr;
  } else {
    cgen_state_.reset();
    context_.reset(new llvm::LLVMContext());
    cgen_state_.reset(
        new CgenState({}, false, false, getExtensionModuleContext(), getContext()));
  }
}

void Executor::update_extension_modules(bool update_runtime_modules_only) {
  auto read_module = [&](ExtModuleKinds module_kind, const std::string& source) {
    /*
      source can be either a filename of a LLVM IR
      or LLVM BC source, or a string containing
      LLVM IR code.
     */
    CHECK(!source.empty());
    switch (module_kind) {
      case ExtModuleKinds::template_module:
      case ExtModuleKinds::l0_template_module:
      case ExtModuleKinds::spirv_helper_funcs_module:
      case ExtModuleKinds::rt_libdevice_module: {
        return read_llvm_module_from_bc_file(source, getContext());
      }
      case ExtModuleKinds::udf_cpu_module: {
        return read_llvm_module_from_ir_file(source, getContext(), /**is_gpu=*/false);
      }
      case ExtModuleKinds::udf_gpu_module: {
        return read_llvm_module_from_ir_file(source, getContext(), /**is_gpu=*/true);
      }
      case ExtModuleKinds::rt_udf_cpu_module: {
        return read_llvm_module_from_ir_string(source, getContext(), /**is_gpu=*/false);
      }
      case ExtModuleKinds::rt_udf_gpu_module: {
        return read_llvm_module_from_ir_string(source, getContext(), /**is_gpu=*/true);
      }
      default: {
        UNREACHABLE();
        return std::unique_ptr<llvm::Module>();
      }
    }
  };
  auto patch_l0_module = [](ExtModuleKinds kind, llvm::Module* m) {
    if (kind == ExtModuleKinds::l0_template_module) {
      m->setTargetTriple("spir64-unknown-unknown");
    }
  };
  auto update_module = [&](ExtModuleKinds module_kind, bool erase_not_found = false) {
    CHECK(extension_module_context_);
    auto& extension_modules = extension_module_context_->getExtensionModules();

    auto it = Executor::extension_module_sources.find(module_kind);
    if (it != Executor::extension_module_sources.end()) {
      auto llvm_module = read_module(module_kind, it->second);
      if (llvm_module) {
        patch_l0_module(module_kind, llvm_module.get());
        extension_modules[module_kind] = std::move(llvm_module);
      } else if (erase_not_found) {
        extension_modules.erase(module_kind);
      } else {
        if (extension_modules.find(module_kind) == extension_modules.end()) {
          LOG(WARNING) << "Failed to update " << ::toString(module_kind)
                       << " LLVM module. The module will be unavailable.";
        } else {
          LOG(WARNING) << "Failed to update " << ::toString(module_kind)
                       << " LLVM module. Using the existing module.";
        }
      }
    } else {
      if (erase_not_found) {
        extension_modules.erase(module_kind);
      } else {
        if (extension_modules.find(module_kind) == extension_modules.end()) {
          LOG(WARNING) << "Source of " << ::toString(module_kind)
                       << " LLVM module is unavailable. The module will be unavailable.";
        } else {
          LOG(WARNING) << "Source of " << ::toString(module_kind)
                       << " LLVM module is unavailable. Using the existing module.";
        }
      }
    }
  };

  if (!update_runtime_modules_only) {
    // required compile-time modules, their requirements are enforced
    // by Executor::initialize_extension_module_sources():
    update_module(ExtModuleKinds::template_module);
    // load-time modules, these are optional:
    update_module(ExtModuleKinds::udf_cpu_module, true);
#ifdef HAVE_CUDA
    update_module(ExtModuleKinds::udf_gpu_module, true);
    update_module(ExtModuleKinds::rt_libdevice_module);
#endif
#ifdef HAVE_L0
    update_module(ExtModuleKinds::l0_template_module);
    update_module(ExtModuleKinds::spirv_helper_funcs_module);
#endif
  }
  // run-time modules, these are optional and erasable:
  update_module(ExtModuleKinds::rt_udf_cpu_module, true);
#ifdef HAVE_CUDA
  update_module(ExtModuleKinds::rt_udf_gpu_module, true);
#endif
}

// Used by StubGenerator::generateStub
Executor::CgenStateManager::CgenStateManager(Executor& executor)
    : executor_(executor)
    , lock_queue_clock_(timer_start())
    , lock_(executor_.compilation_mutex_)
    , cgen_state_(std::move(executor_.cgen_state_))  // store old CgenState instance
{
  executor_.compilation_queue_time_ms_ += timer_stop(lock_queue_clock_);
  executor_.cgen_state_.reset(
      new CgenState(0,
                    false,
                    executor.getConfig().debug.enable_automatic_ir_metadata,
                    executor.getExtensionModuleContext(),
                    executor.getContext()));
}

Executor::CgenStateManager::CgenStateManager(
    Executor& executor,
    const bool allow_lazy_fetch,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit* ra_exe_unit)
    : executor_(executor)
    , lock_queue_clock_(timer_start())
    , lock_(executor_.compilation_mutex_)
    , cgen_state_(std::move(executor_.cgen_state_))  // store old CgenState instance
{
  executor_.compilation_queue_time_ms_ += timer_stop(lock_queue_clock_);
  // nukeOldState creates new CgenState and PlanState instances for
  // the subsequent code generation.  It also resets
  // kernel_queue_time_ms_ and compilation_queue_time_ms_ that we do
  // not currently restore.. should we accumulate these timings?
  executor_.nukeOldState(allow_lazy_fetch, query_infos, ra_exe_unit);
}

Executor::CgenStateManager::~CgenStateManager() {
  // prevent memory leak from hoisted literals
  for (auto& p : executor_.cgen_state_->row_func_hoisted_literals_) {
    auto inst = llvm::dyn_cast<llvm::LoadInst>(p.first);
    if (inst && inst->getNumUses() == 0 && inst->getParent() == nullptr) {
      // The llvm::Value instance stored in p.first is created by the
      // CodeGenerator::codegenHoistedConstantsPlaceholders method.
      p.first->deleteValue();
    }
  }
  executor_.cgen_state_->row_func_hoisted_literals_.clear();

  // move generated StringDictionaryTranslationMgrs and InValueBitmaps
  // to the old CgenState instance as the execution of the generated
  // code uses these bitmaps

  for (auto& str_dict_translation_mgr :
       executor_.cgen_state_->str_dict_translation_mgrs_) {
    cgen_state_->moveStringDictionaryTranslationMgr(std::move(str_dict_translation_mgr));
  }
  executor_.cgen_state_->str_dict_translation_mgrs_.clear();

  for (auto& bm : executor_.cgen_state_->in_values_bitmaps_) {
    cgen_state_->moveInValuesBitmap(bm);
  }
  executor_.cgen_state_->in_values_bitmaps_.clear();

  // restore the old CgenState instance
  executor_.cgen_state_.reset(cgen_state_.release());
}

std::shared_ptr<Executor> Executor::getExecutor(Data_Namespace::DataMgr* data_mgr,
                                                ConfigPtr config,
                                                const std::string& debug_dir,
                                                const std::string& debug_file) {
  INJECT_TIMER(getExecutor);

  if (!config) {
    config = std::make_shared<Config>();
  }

  return std::make_shared<Executor>(
      executor_id_ctr_++, data_mgr, config, debug_dir, debug_file);
}

void Executor::clearMemory(const Data_Namespace::MemoryLevel memory_level,
                           Data_Namespace::DataMgr* data_mgr) {
  switch (memory_level) {
    case Data_Namespace::MemoryLevel::CPU_LEVEL:
    case Data_Namespace::MemoryLevel::GPU_LEVEL: {
      mapd_unique_lock<mapd_shared_mutex> flush_lock(
          execute_mutex_);  // Don't flush memory while queries are running

      CHECK(data_mgr);
      data_mgr->clearMemory(memory_level);
      if (memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
        // The hash table cache uses CPU memory not managed by the buffer manager. In the
        // future, we should manage these allocations with the buffer manager directly.
        // For now, assume the user wants to purge the hash table cache when they clear
        // CPU memory (currently used in ExecuteTest to lower memory pressure)
        JoinHashTableCacheInvalidator::invalidateCaches();
      }
      break;
    }
    default: {
      throw std::runtime_error(
          "Clearing memory levels other than the CPU level or GPU level is not "
          "supported.");
    }
  }
}

size_t Executor::getArenaBlockSize() {
  return g_is_test_env ? 100000000 : (1UL << 32) + kArenaBlockOverhead;
}

StringDictionary* Executor::getStringDictionaryProxy(
    const int dict_id_in,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  const int64_t generation =
      with_generation ? string_dictionary_generations_.getGeneration(dict_id_in) : -1;
  return row_set_mem_owner->getOrAddStringDictProxy(dict_id_in, generation);
}

const std::vector<int32_t>* Executor::getStringProxyTranslationMap(
    const int source_dict_id,
    const int dest_dict_id,
    const RowSetMemoryOwner::StringTranslationType translation_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool with_generation) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  const int64_t source_generation =
      with_generation ? string_dictionary_generations_.getGeneration(source_dict_id) : -1;
  const int64_t dest_generation =
      with_generation ? string_dictionary_generations_.getGeneration(dest_dict_id) : -1;
  return row_set_mem_owner->getOrAddStringProxyTranslationMap(
      source_dict_id, source_generation, dest_dict_id, dest_generation, translation_type);
}

const std::vector<int32_t>* Executor::getIntersectionStringProxyTranslationMap(
    const StringDictionary* source_proxy,
    const StringDictionary* dest_proxy,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const {
  CHECK(row_set_mem_owner);
  std::lock_guard<std::mutex> lock(
      str_dict_mutex_);  // TODO: can we use RowSetMemOwner state mutex here?
  return row_set_mem_owner->addStringProxyIntersectionTranslationMap(source_proxy,
                                                                     dest_proxy);
}

const std::vector<int32_t>* RowSetMemoryOwner::getOrAddStringProxyTranslationMap(
    const int source_dict_id_in,
    const int64_t source_generation,
    const int dest_dict_id_in,
    const int64_t dest_generation,
    const RowSetMemoryOwner::StringTranslationType translation_type) {
  const auto source_proxy = getOrAddStringDictProxy(source_dict_id_in, source_generation);
  auto dest_proxy = getOrAddStringDictProxy(dest_dict_id_in, dest_generation);
  if (translation_type == RowSetMemoryOwner::StringTranslationType::SOURCE_INTERSECTION) {
    return addStringProxyIntersectionTranslationMap(source_proxy, dest_proxy);
  } else {
    return addStringProxyUnionTranslationMap(source_proxy, dest_proxy);
  }
}

bool Executor::isCPUOnly() const {
  CHECK(data_mgr_);
  return !data_mgr_->getCudaMgr();
}

const std::shared_ptr<RowSetMemoryOwner> Executor::getRowSetMemoryOwner() const {
  return row_set_mem_owner_;
}

const TemporaryTables* Executor::getTemporaryTables() const {
  return temporary_tables_;
}

hdk::ResultSetTableTokenPtr Executor::getTemporaryTable(int table_id) const {
  return get_temporary_table(temporary_tables_, table_id);
}

std::shared_ptr<const TableFragmentsInfo> Executor::getTableInfo(
    const int db_id,
    const int table_id) const {
  return input_table_info_cache_.getTableInfo(db_id, table_id);
}

const TableGeneration& Executor::getTableGeneration(int db_id, int table_id) const {
  return table_generations_.getGeneration(db_id, table_id);
}

ExpressionRange Executor::getColRange(const PhysicalInput& phys_input) const {
  return agg_col_range_cache_.getColRange(phys_input);
}

size_t Executor::getNumBytesForFetchedRow(const std::set<int>& table_ids_to_fetch) const {
  size_t num_bytes = 0;
  if (!plan_state_) {
    return 0;
  }
  for (const auto& fetched_col : plan_state_->columns_to_fetch_) {
    int table_id = fetched_col.getTableId();
    if (table_ids_to_fetch.count(table_id) == 0) {
      continue;
    }

    const auto sz = fetched_col.type()->size();
    if (sz < 0) {
      // for varlen types, only account for the pointer/size for each row, for now
      num_bytes += 16;
    } else {
      num_bytes += sz;
    }
  }
  return num_bytes;
}

bool Executor::hasLazyFetchColumns(
    const std::vector<const hdk::ir::Expr*>& target_exprs) const {
  CHECK(plan_state_);
  for (const auto target_expr : target_exprs) {
    if (plan_state_->isLazyFetchColumn(target_expr)) {
      return true;
    }
  }
  return false;
}

std::vector<ColumnLazyFetchInfo> Executor::getColLazyFetchInfo(
    const std::vector<const hdk::ir::Expr*>& target_exprs) const {
  CHECK(plan_state_);
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{false, -1, nullptr});
    } else {
      const auto col_var = dynamic_cast<const hdk::ir::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto local_col_id = plan_state_->getLocalColumnId(col_var, false);
      auto col_type = col_var->type();
      col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_type});
    }
  }
  return col_lazy_fetch_info;
}

void Executor::clearMetaInfoCache() {
  input_table_info_cache_.clear();
  agg_col_range_cache_.clear();
  table_generations_.clear();
}

std::vector<int8_t> Executor::serializeLiterals(
    const std::unordered_map<int, CgenState::LiteralValues>& literals,
    const int device_id) {
  if (literals.empty()) {
    return {};
  }
  const auto dev_literals_it = literals.find(device_id);
  CHECK(dev_literals_it != literals.end());
  const auto& dev_literals = dev_literals_it->second;
  size_t lit_buf_size{0};
  std::vector<std::string> real_strings;
  std::vector<std::vector<double>> double_array_literals;
  std::vector<std::vector<int8_t>> align64_int8_array_literals;
  std::vector<std::vector<int32_t>> int32_array_literals;
  std::vector<std::vector<int8_t>> align32_int8_array_literals;
  std::vector<std::vector<int8_t>> int8_array_literals;
  for (const auto& lit : dev_literals) {
    lit_buf_size = CgenState::addAligned(lit_buf_size, CgenState::literalBytes(lit));
    if (lit.which() == 7) {
      const auto p = boost::get<std::string>(&lit);
      CHECK(p);
      real_strings.push_back(*p);
    } else if (lit.which() == 8) {
      const auto p = boost::get<std::vector<double>>(&lit);
      CHECK(p);
      double_array_literals.push_back(*p);
    } else if (lit.which() == 9) {
      const auto p = boost::get<std::vector<int32_t>>(&lit);
      CHECK(p);
      int32_array_literals.push_back(*p);
    } else if (lit.which() == 10) {
      const auto p = boost::get<std::vector<int8_t>>(&lit);
      CHECK(p);
      int8_array_literals.push_back(*p);
    } else if (lit.which() == 11) {
      const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
      CHECK(p);
      if (p->second == 64) {
        align64_int8_array_literals.push_back(p->first);
      } else if (p->second == 32) {
        align32_int8_array_literals.push_back(p->first);
      } else {
        CHECK(false);
      }
    }
  }
  if (lit_buf_size > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    throw TooManyLiterals();
  }
  int16_t crt_real_str_off = lit_buf_size;
  for (const auto& real_str : real_strings) {
    CHECK_LE(real_str.size(), static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += real_str.size();
  }
  if (double_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(double));
  }
  int16_t crt_double_arr_lit_off = lit_buf_size;
  for (const auto& double_array_literal : double_array_literals) {
    CHECK_LE(double_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += double_array_literal.size() * sizeof(double);
  }
  if (align64_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(uint64_t));
  }
  int16_t crt_align64_int8_arr_lit_off = lit_buf_size;
  for (const auto& align64_int8_array_literal : align64_int8_array_literals) {
    CHECK_LE(align64_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align64_int8_array_literal.size();
  }
  if (int32_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_int32_arr_lit_off = lit_buf_size;
  for (const auto& int32_array_literal : int32_array_literals) {
    CHECK_LE(int32_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int32_array_literal.size() * sizeof(int32_t);
  }
  if (align32_int8_array_literals.size() > 0) {
    lit_buf_size = align(lit_buf_size, sizeof(int32_t));
  }
  int16_t crt_align32_int8_arr_lit_off = lit_buf_size;
  for (const auto& align32_int8_array_literal : align32_int8_array_literals) {
    CHECK_LE(align32_int8_array_literals.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += align32_int8_array_literal.size();
  }
  int16_t crt_int8_arr_lit_off = lit_buf_size;
  for (const auto& int8_array_literal : int8_array_literals) {
    CHECK_LE(int8_array_literal.size(),
             static_cast<size_t>(std::numeric_limits<int16_t>::max()));
    lit_buf_size += int8_array_literal.size();
  }
  unsigned crt_real_str_idx = 0;
  unsigned crt_double_arr_lit_idx = 0;
  unsigned crt_align64_int8_arr_lit_idx = 0;
  unsigned crt_int32_arr_lit_idx = 0;
  unsigned crt_align32_int8_arr_lit_idx = 0;
  unsigned crt_int8_arr_lit_idx = 0;
  std::vector<int8_t> serialized(lit_buf_size);
  size_t off{0};
  for (const auto& lit : dev_literals) {
    const auto lit_bytes = CgenState::literalBytes(lit);
    off = CgenState::addAligned(off, lit_bytes);
    switch (lit.which()) {
      case 0: {
        const auto p = boost::get<int8_t>(&lit);
        CHECK(p);
        serialized[off - lit_bytes] = *p;
        break;
      }
      case 1: {
        const auto p = boost::get<int16_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 2: {
        const auto p = boost::get<int32_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 3: {
        const auto p = boost::get<int64_t>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 4: {
        const auto p = boost::get<float>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 5: {
        const auto p = boost::get<double>(&lit);
        CHECK(p);
        memcpy(&serialized[off - lit_bytes], p, lit_bytes);
        break;
      }
      case 6: {
        const auto p = boost::get<std::pair<std::string, int>>(&lit);
        CHECK(p);
        const auto str_id =
            (config_->exec.enable_experimental_string_functions)
                ? getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                      ->getIdOfString(p->first)
                : getStringDictionaryProxy(p->second, row_set_mem_owner_, true)
                      ->getOrAdd(p->first);
        memcpy(&serialized[off - lit_bytes], &str_id, lit_bytes);
        break;
      }
      case 7: {
        const auto p = boost::get<std::string>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_real_str_off << 16;
        const auto& crt_real_str = real_strings[crt_real_str_idx];
        off_and_len |= static_cast<int16_t>(crt_real_str.size());
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_real_str_off], crt_real_str.data(), crt_real_str.size());
        ++crt_real_str_idx;
        crt_real_str_off += crt_real_str.size();
        break;
      }
      case 8: {
        const auto p = boost::get<std::vector<double>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_double_arr_lit_off << 16;
        const auto& crt_double_arr_lit = double_array_literals[crt_double_arr_lit_idx];
        int32_t len = crt_double_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t double_array_bytesize = len * sizeof(double);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_double_arr_lit_off],
               crt_double_arr_lit.data(),
               double_array_bytesize);
        ++crt_double_arr_lit_idx;
        crt_double_arr_lit_off += double_array_bytesize;
        break;
      }
      case 9: {
        const auto p = boost::get<std::vector<int32_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int32_arr_lit_off << 16;
        const auto& crt_int32_arr_lit = int32_array_literals[crt_int32_arr_lit_idx];
        int32_t len = crt_int32_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int32_array_bytesize = len * sizeof(int32_t);
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int32_arr_lit_off],
               crt_int32_arr_lit.data(),
               int32_array_bytesize);
        ++crt_int32_arr_lit_idx;
        crt_int32_arr_lit_off += int32_array_bytesize;
        break;
      }
      case 10: {
        const auto p = boost::get<std::vector<int8_t>>(&lit);
        CHECK(p);
        int32_t off_and_len = crt_int8_arr_lit_off << 16;
        const auto& crt_int8_arr_lit = int8_array_literals[crt_int8_arr_lit_idx];
        int32_t len = crt_int8_arr_lit.size();
        CHECK_EQ((len >> 16), 0);
        off_and_len |= static_cast<int16_t>(len);
        int32_t int8_array_bytesize = len;
        memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
        memcpy(&serialized[crt_int8_arr_lit_off],
               crt_int8_arr_lit.data(),
               int8_array_bytesize);
        ++crt_int8_arr_lit_idx;
        crt_int8_arr_lit_off += int8_array_bytesize;
        break;
      }
      case 11: {
        const auto p = boost::get<std::pair<std::vector<int8_t>, int>>(&lit);
        CHECK(p);
        if (p->second == 64) {
          int32_t off_and_len = crt_align64_int8_arr_lit_off << 16;
          const auto& crt_align64_int8_arr_lit =
              align64_int8_array_literals[crt_align64_int8_arr_lit_idx];
          int32_t len = crt_align64_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align64_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align64_int8_arr_lit_off],
                 crt_align64_int8_arr_lit.data(),
                 align64_int8_array_bytesize);
          ++crt_align64_int8_arr_lit_idx;
          crt_align64_int8_arr_lit_off += align64_int8_array_bytesize;
        } else if (p->second == 32) {
          int32_t off_and_len = crt_align32_int8_arr_lit_off << 16;
          const auto& crt_align32_int8_arr_lit =
              align32_int8_array_literals[crt_align32_int8_arr_lit_idx];
          int32_t len = crt_align32_int8_arr_lit.size();
          CHECK_EQ((len >> 16), 0);
          off_and_len |= static_cast<int16_t>(len);
          int32_t align32_int8_array_bytesize = len;
          memcpy(&serialized[off - lit_bytes], &off_and_len, lit_bytes);
          memcpy(&serialized[crt_align32_int8_arr_lit_off],
                 crt_align32_int8_arr_lit.data(),
                 align32_int8_array_bytesize);
          ++crt_align32_int8_arr_lit_idx;
          crt_align32_int8_arr_lit_off += align32_int8_array_bytesize;
        } else {
          CHECK(false);
        }
        break;
      }
      default:
        CHECK(false);
    }
  }
  return serialized;
}

int Executor::deviceCount(const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::GPU) {
    return gpuMgr()->getDeviceCount();
  } else {
    return 1;
  }
}

int Executor::deviceCountForMemoryLevel(
    const Data_Namespace::MemoryLevel memory_level) const {
  return memory_level == GPU_LEVEL ? deviceCount(ExecutorDeviceType::GPU)
                                   : deviceCount(ExecutorDeviceType::CPU);
}

CudaMgr_Namespace::CudaMgr* Executor::cudaMgr() const {
  CHECK(data_mgr_);
  auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  return cuda_mgr;
}

GpuMgr* Executor::gpuMgr() const {
  CHECK(data_mgr_);
  auto gpu_mgr = data_mgr_->getGpuMgr();
  CHECK(gpu_mgr);
  return gpu_mgr;
}

bool Executor::deviceSupportsFP64(const ExecutorDeviceType dt) const {
  if (dt == ExecutorDeviceType::GPU) {
    return gpuMgr()->hasFP64Support();
  }
  return true;
}

// TODO(alex): remove or split
std::pair<int64_t, int32_t> Executor::reduceResults(hdk::ir::AggType agg,
                                                    const hdk::ir::Type* type,
                                                    const int64_t agg_init_val,
                                                    const int8_t out_byte_width,
                                                    const int64_t* out_vec,
                                                    const size_t out_vec_sz,
                                                    const bool is_group_by,
                                                    const bool float_argument_input) {
  switch (agg) {
    case hdk::ir::AggType::kAvg:
    case hdk::ir::AggType::kSum:
      if (0 != agg_init_val) {
        if (type->isInteger() || type->isDecimal() || type->isDateTime() ||
            type->isBoolean()) {
          int64_t agg_result = agg_init_val;
          for (size_t i = 0; i < out_vec_sz; ++i) {
            agg_sum_skip_val(&agg_result, out_vec[i], agg_init_val);
          }
          return {agg_result, 0};
        } else {
          CHECK(type->isFloatingPoint());
          switch (out_byte_width) {
            case 4: {
              int agg_result = static_cast<int32_t>(agg_init_val);
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_float_skip_val(
                    &agg_result,
                    *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
              }
              const int64_t converted_bin =
                  float_argument_input
                      ? static_cast<int64_t>(agg_result)
                      : float_to_double_bin(static_cast<int32_t>(agg_result), true);
              return {converted_bin, 0};
              break;
            }
            case 8: {
              int64_t agg_result = agg_init_val;
              for (size_t i = 0; i < out_vec_sz; ++i) {
                agg_sum_double_skip_val(
                    &agg_result,
                    *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                    *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
              }
              return {agg_result, 0};
              break;
            }
            default:
              CHECK(false);
          }
        }
      }
      if (type->isInteger() || type->isDecimal() || type->isDateTime()) {
        int64_t agg_result = 0;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_result += out_vec[i];
        }
        return {agg_result, 0};
      } else {
        CHECK(type->isFloatingPoint());
        switch (out_byte_width) {
          case 4: {
            float r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i]));
            }
            const auto float_bin = *reinterpret_cast<const int32_t*>(may_alias_ptr(&r));
            const int64_t converted_bin =
                float_argument_input ? float_bin : float_to_double_bin(float_bin, true);
            return {converted_bin, 0};
          }
          case 8: {
            double r = 0.;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              r += *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i]));
            }
            return {*reinterpret_cast<const int64_t*>(may_alias_ptr(&r)), 0};
          }
          default:
            CHECK(false);
        }
      }
      break;
    case hdk::ir::AggType::kCount: {
      uint64_t agg_result = 0;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        const uint64_t out = static_cast<uint64_t>(out_vec[i]);
        agg_result += out;
      }
      return {static_cast<int64_t>(agg_result), 0};
    }
    case hdk::ir::AggType::kMin: {
      if (type->isInteger() || type->isDecimal() || type->isDateTime() ||
          type->isBoolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_min_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input
                    ? static_cast<int64_t>(agg_result)
                    : float_to_double_bin(static_cast<int32_t>(agg_result), true);
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_min_double_skip_val(
                  &agg_result,
                  *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    }
    case hdk::ir::AggType::kMax:
      if (type->isInteger() || type->isDecimal() || type->isDateTime() ||
          type->isBoolean()) {
        int64_t agg_result = agg_init_val;
        for (size_t i = 0; i < out_vec_sz; ++i) {
          agg_max_skip_val(&agg_result, out_vec[i], agg_init_val);
        }
        return {agg_result, 0};
      } else {
        switch (out_byte_width) {
          case 4: {
            int32_t agg_result = static_cast<int32_t>(agg_init_val);
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_float_skip_val(
                  &agg_result,
                  *reinterpret_cast<const float*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const float*>(may_alias_ptr(&agg_init_val)));
            }
            const int64_t converted_bin =
                float_argument_input ? static_cast<int64_t>(agg_result)
                                     : float_to_double_bin(agg_result, type->nullable());
            return {converted_bin, 0};
          }
          case 8: {
            int64_t agg_result = agg_init_val;
            for (size_t i = 0; i < out_vec_sz; ++i) {
              agg_max_double_skip_val(
                  &agg_result,
                  *reinterpret_cast<const double*>(may_alias_ptr(&out_vec[i])),
                  *reinterpret_cast<const double*>(may_alias_ptr(&agg_init_val)));
            }
            return {agg_result, 0};
          }
          default:
            CHECK(false);
        }
      }
    case hdk::ir::AggType::kSingleValue: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          if (agg_result == agg_init_val) {
            agg_result = out_vec[i];
          } else if (out_vec[i] != agg_result) {
            return {agg_result, Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES};
          }
        }
      }
      return {agg_result, 0};
    }
    case hdk::ir::AggType::kSample: {
      int64_t agg_result = agg_init_val;
      for (size_t i = 0; i < out_vec_sz; ++i) {
        if (out_vec[i] != agg_init_val) {
          agg_result = out_vec[i];
          break;
        }
      }
      return {agg_result, 0};
    }
    default:
      CHECK(false);
  }
  abort();
}

namespace {

hdk::ResultSetTable get_merged_result(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  auto& first = results_per_device.front().first;
  CHECK(first);
  for (size_t dev_idx = 1; dev_idx < results_per_device.size(); ++dev_idx) {
    const auto& next = results_per_device[dev_idx].first;
    CHECK(next);
    first->append(*next);
  }
  return hdk::ResultSetTable(std::move(first));
}

hdk::ResultSetTable get_separate_results(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  std::vector<ResultSetPtr> results;
  results.reserve(results_per_device.size());
  for (auto& r : results_per_device) {
    results.emplace_back(r.first);
  }
  return hdk::ResultSetTable(std::move(results));
}

}  // namespace

hdk::ResultSetTable Executor::resultsUnion(SharedKernelContext& shared_context,
                                           const RelAlgExecutionUnit& ra_exe_unit,
                                           bool merge,
                                           bool sort_by_table_id,
                                           const std::map<int, size_t>& order_map) {
  auto timer = DEBUG_TIMER(__func__);
  auto& results_per_device = shared_context.getFragmentResults();
  if (results_per_device.empty()) {
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(
          get_target_info(target_expr, getConfig().exec.group_by.bigint_count));
    }
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       row_set_mem_owner_,
                                       data_mgr_,
                                       blockSize(),
                                       gridSize());
  }
  using IndexedResultSet = std::pair<ResultSetPtr, std::vector<size_t>>;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [sort_by_table_id, &order_map](const IndexedResultSet& lhs,
                                           const IndexedResultSet& rhs) {
              CHECK_GE(lhs.second.size(), size_t(1));
              CHECK_GE(rhs.second.size(), size_t(1));
              if (sort_by_table_id) {
                auto ltid = lhs.first->getOuterTableId();
                auto rtid = rhs.first->getOuterTableId();
                if (ltid != rtid) {
                  return order_map.at(ltid) < order_map.at(rtid);
                }
              }
              return lhs.second.front() < rhs.second.front();
            });

  if (merge) {
    return get_merged_result(results_per_device);
  }
  return get_separate_results(results_per_device);
}

ResultSetPtr Executor::reduceMultiDeviceResults(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  if (ra_exe_unit.estimator) {
    return reduce_estimator_results(ra_exe_unit, results_per_device);
  }

  if (results_per_device.empty()) {
    std::vector<TargetInfo> targets;
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      targets.push_back(
          get_target_info(target_expr, getConfig().exec.group_by.bigint_count));
    }
    return std::make_shared<ResultSet>(targets,
                                       ExecutorDeviceType::CPU,
                                       QueryMemoryDescriptor(),
                                       row_set_mem_owner,
                                       data_mgr_,
                                       blockSize(),
                                       gridSize());
  }

  return reduceMultiDeviceResultSets(
      results_per_device,
      row_set_mem_owner,
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
      co);
}

namespace {

ReductionCode get_reduction_code(
    const Config& config,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    int64_t* compilation_queue_time,
    Executor* executor,
    const CompilationOptions& co) {
  auto clock_begin = timer_start();
  // ResultSetReductionJIT::codegen compilation-locks if new code will be generated
  *compilation_queue_time = timer_stop(clock_begin);
  const auto& this_result_set = results_per_device[0].first;
  ResultSetReductionJIT reduction_jit(this_result_set->getQueryMemDesc(),
                                      this_result_set->getTargetInfos(),
                                      this_result_set->getTargetInitVals(),
                                      config,
                                      executor);
  return reduction_jit.codegen();
};

}  // namespace

bool couldUseParallelReduce(const QueryMemoryDescriptor& desc) {
  if (desc.getQueryDescriptionType() == QueryDescriptionType::NonGroupedAggregate &&
      desc.getCountDistinctDescriptorsSize()) {
    return true;
  }

  if (desc.getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash) {
    return true;
  }

  return false;
}

ResultSetPtr Executor::reduceMultiDeviceResultSets(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<ResultSet> reduced_results;

  const auto& first = results_per_device.front().first;

  if (results_per_device.size() == 1) {
    // This finalization is optional but done here to have finalization time
    // to be a part of reduction.
    first->finalizeAggregates();
    return first;
  }

  if (query_mem_desc.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    const auto total_entry_count = std::accumulate(
        results_per_device.begin(),
        results_per_device.end(),
        size_t(0),
        [](const size_t init, const std::pair<ResultSetPtr, std::vector<size_t>>& rs) {
          const auto& r = rs.first;
          return init + r->getQueryMemDesc().getEntryCount();
        });
    CHECK(total_entry_count);
    auto query_mem_desc = first->getQueryMemDesc();
    query_mem_desc.setEntryCount(total_entry_count);
    reduced_results = std::make_shared<ResultSet>(first->getTargetInfos(),
                                                  ExecutorDeviceType::CPU,
                                                  query_mem_desc,
                                                  row_set_mem_owner,
                                                  data_mgr_,
                                                  blockSize(),
                                                  gridSize());
    auto result_storage = reduced_results->allocateStorage(plan_state_->init_agg_vals_);
    reduced_results->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        ResultSetReduction::moveEntriesToBuffer<int32_t>(
            first->getStorage()->getQueryMemDesc(),
            first->getStorage()->getUnderlyingBuffer(),
            result_storage->getUnderlyingBuffer(),
            query_mem_desc.getEntryCount());
        break;
      case 8:
        ResultSetReduction::moveEntriesToBuffer<int64_t>(
            first->getStorage()->getQueryMemDesc(),
            first->getStorage()->getUnderlyingBuffer(),
            result_storage->getUnderlyingBuffer(),
            query_mem_desc.getEntryCount());
        break;
      default:
        CHECK(false);
    }
  } else {
    reduced_results = first;
    reduced_results->invalidateCachedRowCount();
  }

  int64_t compilation_queue_time = 0;
  const auto reduction_code = get_reduction_code(
      getConfig(), results_per_device, &compilation_queue_time, this, co);

  if (couldUseParallelReduce(query_mem_desc)) {
    std::vector<ResultSetStorage*> storages;
    for (auto& rs : results_per_device) {
      storages.push_back(const_cast<ResultSetStorage*>(rs.first->getStorage()));
    }
    tbb::parallel_reduce(
        tbb::blocked_range(storages.begin(), storages.end()),
        (ResultSetStorage*)nullptr,
        [&](auto r, ResultSetStorage* res) {
          for (auto i = r.begin() + 1; i != r.end(); ++i) {
            ResultSetReduction::reduce(
                **r.begin(), **i, {}, reduction_code, getConfig(), this);
          }
          if (res) {
            ResultSetReduction::reduce(
                *res, *(*r.begin()), {}, reduction_code, getConfig(), this);
            return res;
          }
          return *r.begin();
        },
        [&](ResultSetStorage* lhs, ResultSetStorage* rhs) {
          if (!lhs) {
            return rhs;
          }
          if (!rhs) {
            return lhs;
          }
          ResultSetReduction::reduce(*lhs, *rhs, {}, reduction_code, getConfig(), this);
          return lhs;
        });
  } else {
    for (size_t i = 1; i < results_per_device.size(); ++i) {
      ResultSetReduction::reduce(*reduced_results->getStorage(),
                                 *(results_per_device[i].first->getStorage()),
                                 {},
                                 reduction_code,
                                 getConfig(),
                                 this);
    }
  }
  // This finalization is required because reduced results might reference
  // memory owned by other ResultSets (Quantile aggregate). Finalize aggregates
  // so that we can safely destroy original ResultSets.
  reduced_results->finalizeAggregates();
  reduced_results->addCompilationQueueTime(compilation_queue_time);
  return reduced_results;
}

ResultSetPtr Executor::reduceSpeculativeTopN(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc) const {
  if (results_per_device.size() == 1) {
    return std::move(results_per_device.front().first);
  }
  const auto top_n = ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  SpeculativeTopNMap m;
  for (const auto& result : results_per_device) {
    auto rows = result.first;
    CHECK(rows);
    if (!rows) {
      continue;
    }
    SpeculativeTopNMap that(
        *rows,
        ra_exe_unit.target_exprs,
        std::max(size_t(10000 * std::max(1, static_cast<int>(log(top_n)))), top_n));
    m.reduce(that);
  }
  CHECK_EQ(size_t(1), ra_exe_unit.sort_info.order_entries.size());
  const auto desc = ra_exe_unit.sort_info.order_entries.front().is_desc;
  return m.asRows(ra_exe_unit, row_set_mem_owner, query_mem_desc, this, top_n, desc);
}

hdk::ResultSetTable Executor::reducePartitionHistogram(
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    const QueryMemoryDescriptor& query_mem_desc,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const {
  std::vector<ResultSetPtr> results;
  std::vector<int64_t*> buffers;

  results.reserve(results_per_device.size() + 1);
  buffers.reserve(results_per_device.size() + 1);

  // In the reduction result we want each fragment result to hold an offset in
  // output buffer instead of number of rows. Also, we want to make an additional
  // result set holding final partition sizes. Achieve it by using a new zero-filled
  // buffer for the the first fragment and partial sums for all other having
  // partition sizes in the last of them.
  // Additionally, we want them to be columnar Projection instead of PerfectHash
  // to later zero-copy fetch all rows.
  using IndexedResultSet = std::pair<ResultSetPtr, std::vector<size_t>>;
  std::sort(results_per_device.begin(),
            results_per_device.end(),
            [](const IndexedResultSet& lhs, const IndexedResultSet& rhs) {
              CHECK_GE(lhs.second.size(), size_t(1));
              CHECK_GE(rhs.second.size(), size_t(1));
              return lhs.second.front() < rhs.second.front();
            });

  auto parts = query_mem_desc.getEntryCount();
  auto proj_mem_desc = query_mem_desc;
  proj_mem_desc.setQueryDescriptionType(QueryDescriptionType::Projection);
  proj_mem_desc.setHasKeylessHash(false);
  proj_mem_desc.clearGroupColWidths();
  // Currently, all perfect hash tables are expected to to use 8 byte padded width.
  CHECK_EQ(static_cast<int>(proj_mem_desc.getPaddedSlotWidthBytes(0)), 8);
  auto first_rs =
      std::make_shared<ResultSet>(results_per_device.front().first->getTargetInfos(),
                                  ExecutorDeviceType::CPU,
                                  proj_mem_desc,
                                  row_set_mem_owner,
                                  data_mgr_,
                                  blockSize(),
                                  gridSize());
  first_rs->allocateStorage(plan_state_->init_agg_vals_);
  results.push_back(first_rs);
  buffers.push_back(
      reinterpret_cast<int64_t*>(first_rs->getStorage()->getUnderlyingBuffer()));
  for (auto& pr : results_per_device) {
    auto buf = pr.first->getStorage()->getUnderlyingBuffer();
    auto proj_rs =
        std::make_shared<ResultSet>(results_per_device.front().first->getTargetInfos(),
                                    ExecutorDeviceType::CPU,
                                    proj_mem_desc,
                                    row_set_mem_owner,
                                    data_mgr_,
                                    blockSize(),
                                    gridSize());
    proj_rs->allocateStorage(buf, {});
    results.push_back(proj_rs);
    buffers.push_back(reinterpret_cast<int64_t*>(buf));
  }

  memset(buffers[0], 0, sizeof(int64_t) * parts);
  for (size_t i = 2; i < buffers.size(); ++i) {
    for (size_t j = 0; j < parts; ++j) {
      buffers[i][j] += buffers[i - 1][j];
    }
  }

  return hdk::ResultSetTable(std::move(results));
}

namespace {

std::unordered_set<int> get_available_gpus(const Data_Namespace::DataMgr* data_mgr) {
  CHECK(data_mgr);
  std::unordered_set<int> available_gpus;
  if (data_mgr->gpusPresent()) {
    CHECK(data_mgr->getGpuMgr());
    const int gpu_count = data_mgr->getGpuMgr()->getDeviceCount();
    CHECK_GT(gpu_count, 0);
    for (int gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
      available_gpus.insert(gpu_id);
    }
  }
  return available_gpus;
}

// Compute a very conservative entry count for the output buffer entry count using no
// other information than the number of tuples in each table and multiplying them
// together.
size_t compute_buffer_entry_guess(const std::vector<InputTableInfo>& query_infos) {
  // Check for overflows since we're multiplying potentially big table sizes.
  using checked_size_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::unsigned_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;
  checked_size_t max_groups_buffer_entry_guess = 1;
  for (const auto& query_info : query_infos) {
    CHECK(!query_info.info->fragments.empty());
    auto it = std::max_element(query_info.info->fragments.begin(),
                               query_info.info->fragments.end(),
                               [](const FragmentInfo& f1, const FragmentInfo& f2) {
                                 return f1.getNumTuples() < f2.getNumTuples();
                               });
    max_groups_buffer_entry_guess *= it->getNumTuples();
  }
  // Cap the rough approximation to 100M entries, it's unlikely we can do a great job for
  // baseline group layout with that many entries anyway.
  constexpr size_t max_groups_buffer_entry_guess_cap = 100000000;
  try {
    return std::min(static_cast<size_t>(max_groups_buffer_entry_guess),
                    max_groups_buffer_entry_guess_cap);
  } catch (...) {
    return max_groups_buffer_entry_guess_cap;
  }
}

std::string get_table_name(const InputDescriptor& input_desc,
                           const SchemaProvider& schema_provider) {
  const auto tinfo =
      schema_provider.getTableInfo(input_desc.getDatabaseId(), input_desc.getTableId());
  CHECK(tinfo);
  return tinfo->name;
}

inline size_t getDeviceBasedScanLimit(const ExecutorDeviceType device_type,
                                      const int device_count) {
  if (device_type == ExecutorDeviceType::GPU) {
    return device_count * Executor::high_scan_limit;
  }
  return Executor::high_scan_limit;
}

void checkWorkUnitWatchdog(const RelAlgExecutionUnit& ra_exe_unit,
                           const std::vector<InputTableInfo>& table_infos,
                           const SchemaProvider& schema_provider,
                           const ExecutorDeviceType device_type,
                           const int device_count) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const hdk::ir::AggExpr*>(target_expr)) {
      return;
    }
  }
  if (!ra_exe_unit.scan_limit && table_infos.size() == 1 &&
      table_infos.front().info->getPhysicalNumTuples() < Executor::high_scan_limit) {
    // Allow a query with no scan limit to run on small tables
    return;
  }
  if (ra_exe_unit.sort_info.algorithm != SortAlgorithm::StreamingTopN &&
      ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit ||
       ra_exe_unit.scan_limit > getDeviceBasedScanLimit(device_type, device_count))) {
    std::vector<std::string> table_names;
    const auto& input_descs = ra_exe_unit.input_descs;
    for (const auto& input_desc : input_descs) {
      table_names.push_back(get_table_name(input_desc, schema_provider));
    }
    if (!ra_exe_unit.scan_limit) {
      throw WatchdogException(
          "Projection query would require a scan without a limit on table(s): " +
          boost::algorithm::join(table_names, ", "));
    } else {
      throw WatchdogException(
          "Projection query output result set on table(s): " +
          boost::algorithm::join(table_names, ", ") + "  would contain " +
          std::to_string(ra_exe_unit.scan_limit) +
          " rows, which is more than the current system limit of " +
          std::to_string(getDeviceBasedScanLimit(device_type, device_count)));
    }
  }
}

}  // namespace

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          unsigned trivial_loop_join_threshold) {
  if (ra_exe_unit.input_descs.size() < 2) {
    return false;
  }

  // We only support loop join at the end of folded joins
  // where ra_exe_unit.input_descs.size() > 2 for now.
  const auto inner_table_id = ra_exe_unit.input_descs.back().getTableId();

  std::optional<size_t> inner_table_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (query_infos[i].table_id == inner_table_id) {
      inner_table_idx = i;
      break;
    }
  }
  CHECK(inner_table_idx);
  return query_infos[*inner_table_idx].info->getNumTuples() <=
         trivial_loop_join_threshold;
}

namespace {

template <typename T>
std::vector<std::string> expr_container_to_string(const T& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    if (!expr) {
      expr_strs.emplace_back("NULL");
    } else {
      expr_strs.emplace_back(expr->toString());
    }
  }
  return expr_strs;
}

template <>
std::vector<std::string> expr_container_to_string(
    const std::list<hdk::ir::OrderEntry>& expr_container) {
  std::vector<std::string> expr_strs;
  for (const auto& expr : expr_container) {
    expr_strs.emplace_back(expr.toString());
  }
  return expr_strs;
}

std::string sort_algorithm_to_string(const SortAlgorithm algorithm) {
  switch (algorithm) {
    case SortAlgorithm::Default:
      return "ResultSet";
    case SortAlgorithm::SpeculativeTopN:
      return "Speculative Top N";
    case SortAlgorithm::StreamingTopN:
      return "Streaming Top N";
  }
  UNREACHABLE();
  return "";
}

}  // namespace

std::string ra_exec_unit_desc_for_caching(const RelAlgExecutionUnit& ra_exe_unit) {
  // todo(yoonmin): replace a cache key as a DAG representation of a query plan
  // instead of ra_exec_unit description if possible
  std::ostringstream os;
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    os << input_col_desc->getTableId() << "," << input_col_desc->getColId() << ","
       << input_col_desc->getNestLevel();
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    for (const auto& qual : ra_exe_unit.simple_quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.quals.empty()) {
    for (const auto& qual : ra_exe_unit.quals) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  if (!ra_exe_unit.join_quals.empty()) {
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << std::to_string(i) << ::toString(join_condition.type);
      for (const auto& qual : join_condition.quals) {
        if (qual) {
          os << qual->toString() << ",";
        }
      }
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    for (const auto& qual : ra_exe_unit.groupby_exprs) {
      if (qual) {
        os << qual->toString() << ",";
      }
    }
  }
  for (const auto& expr : ra_exe_unit.target_exprs) {
    if (expr) {
      os << expr->toString() << ",";
    }
  }
  os << ::toString(ra_exe_unit.estimator == nullptr);
  os << std::to_string(ra_exe_unit.scan_limit);
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit) {
  auto query_plan_dag =
      ra_exe_unit.query_plan_dag == EMPTY_QUERY_PLAN ? "N/A" : ra_exe_unit.query_plan_dag;
  os << "\n\tExtracted Query Plan Dag: " << query_plan_dag;
  os << "\n\tTable/Col/Levels: ";
  for (const auto& input_col_desc : ra_exe_unit.input_col_descs) {
    os << "(" << input_col_desc->getTableId() << ", " << input_col_desc->getColId()
       << ", " << input_col_desc->getNestLevel() << ") ";
  }
  if (!ra_exe_unit.simple_quals.empty()) {
    os << "\n\tSimple Quals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.simple_quals),
                                 ", ");
  }
  if (!ra_exe_unit.quals.empty()) {
    os << "\n\tQuals: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.quals), ", ");
  }
  if (!ra_exe_unit.join_quals.empty()) {
    os << "\n\tJoin Quals: ";
    for (size_t i = 0; i < ra_exe_unit.join_quals.size(); i++) {
      const auto& join_condition = ra_exe_unit.join_quals[i];
      os << "\t\t" << std::to_string(i) << " " << ::toString(join_condition.type);
      os << boost::algorithm::join(expr_container_to_string(join_condition.quals), ", ");
    }
  }
  if (!ra_exe_unit.groupby_exprs.empty()) {
    os << "\n\tGroup By: "
       << boost::algorithm::join(expr_container_to_string(ra_exe_unit.groupby_exprs),
                                 ", ");
  }
  os << "\n\tProjected targets: "
     << boost::algorithm::join(expr_container_to_string(ra_exe_unit.target_exprs), ", ");
  os << "\n\tHas Estimator: " << ::toString(ra_exe_unit.estimator == nullptr);
  os << "\n\tSort Info: ";
  const auto& sort_info = ra_exe_unit.sort_info;
  os << "\n\t  Order Entries: "
     << boost::algorithm::join(expr_container_to_string(sort_info.order_entries), ", ");
  os << "\n\t  Algorithm: " << sort_algorithm_to_string(sort_info.algorithm);
  os << "\n\t  Limit: " << std::to_string(sort_info.limit);
  os << "\n\t  Offset: " << std::to_string(sort_info.offset);
  os << "\n\tScan Limit: " << std::to_string(ra_exe_unit.scan_limit);
  if (ra_exe_unit.union_all) {
    os << "\n\tUnion: " << std::string(*ra_exe_unit.union_all ? "UNION ALL" : "UNION");
  }
  if (ra_exe_unit.shuffle_fn) {
    os << "\n\tShuffle: " << *ra_exe_unit.shuffle_fn;
  }
  if (ra_exe_unit.partition_offsets_col) {
    os << "\n\tPartition offsets column: "
       << ra_exe_unit.partition_offsets_col->toString();
  }
  os << "\n\tPartitioned aggregation: " << ra_exe_unit.partitioned_aggregation;

  return os;
}

namespace {

RelAlgExecutionUnit replace_scan_limit(const RelAlgExecutionUnit& ra_exe_unit_in,
                                       const size_t new_scan_limit) {
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          ra_exe_unit_in.groupby_exprs,
          ra_exe_unit_in.target_exprs,
          ra_exe_unit_in.estimator,
          ra_exe_unit_in.sort_info,
          new_scan_limit,
          ra_exe_unit_in.query_plan_dag,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map,
          ra_exe_unit_in.union_all,
          ra_exe_unit_in.shuffle_fn,
          ra_exe_unit_in.partition_offsets_col,
          ra_exe_unit_in.partitioned_aggregation,
          ra_exe_unit_in.cost_model};
}

}  // namespace

hdk::ResultSetTable Executor::executeWorkUnit(
    size_t& max_groups_buffer_entry_guess,
    const bool is_agg,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const bool has_cardinality_estimation,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache) {
  VLOG(1) << "Executor " << executor_id_ << " is executing work unit:" << ra_exe_unit_in;

  ScopeGuard cleanup_post_execution = [this] {
    // cleanup/unpin GPU buffer allocations
    // TODO: separate out this state into a single object
    plan_state_.reset(nullptr);
    if (cgen_state_) {
      cgen_state_->in_values_bitmaps_.clear();
    }
  };

  bool has_proj_unnest =
      !UnnestedVarsCollector::collect(ra_exe_unit_in.target_exprs).empty();
  try {
    auto result = executeWorkUnitImpl(max_groups_buffer_entry_guess,
                                      is_agg,
                                      !has_proj_unnest,
                                      query_infos,
                                      ra_exe_unit_in,
                                      co,
                                      eo,
                                      row_set_mem_owner_,
                                      has_cardinality_estimation,
                                      data_provider,
                                      column_cache);
    result.setKernelQueueTime(kernel_queue_time_ms_);
    result.addCompilationQueueTime(compilation_queue_time_ms_);
    if (eo.just_validate) {
      result.setValidationOnlyRes();
    }
    return result;
  } catch (const CompilationRetryNewScanLimit& e) {
    CHECK(!ra_exe_unit_in.shuffle_fn);
    auto result =
        executeWorkUnitImpl(max_groups_buffer_entry_guess,
                            is_agg,
                            false,
                            query_infos,
                            replace_scan_limit(ra_exe_unit_in, e.new_scan_limit_),
                            co,
                            eo,
                            row_set_mem_owner_,
                            has_cardinality_estimation,
                            data_provider,
                            column_cache);
    result.setKernelQueueTime(kernel_queue_time_ms_);
    result.addCompilationQueueTime(compilation_queue_time_ms_);
    if (eo.just_validate) {
      result.setValidationOnlyRes();
    }
    return result;
  }
}

std::shared_ptr<StreamExecutionContext> Executor::prepareStreamingExecution(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const std::vector<InputTableInfo>& query_infos,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache) {
  const auto device_type = getDeviceTypeForTargets(ra_exe_unit, co.device_type);

  auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
  query_comp_desc_owned->setUseGroupByBufferDesc(co.use_groupby_buffer_desc);
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;

  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};

  auto column_fetcher =
      std::make_unique<ColumnFetcher>(this, data_provider, column_cache);

  query_mem_desc_owned = query_comp_desc_owned->compile(-1,
                                                        crt_min_byte_width,
                                                        false,
                                                        ra_exe_unit,
                                                        query_infos,
                                                        *column_fetcher,
                                                        {device_type,
                                                         co.hoist_literals,
                                                         co.opt_level,
                                                         co.with_dynamic_watchdog,
                                                         co.allow_lazy_fetch,
                                                         co.filter_on_deleted_column,
                                                         co.explain_type,
                                                         co.register_intel_jit_listener},
                                                        eo,
                                                        this);

  for (const auto target_expr : ra_exe_unit.target_exprs) {
    plan_state_->target_exprs_.push_back(target_expr);
  }

  CHECK(query_mem_desc_owned);

  auto ctx = std::make_shared<StreamExecutionContext>(std::move(ra_exe_unit), co, eo);
  ctx->query_comp_desc = std::move(query_comp_desc_owned);
  ctx->query_mem_desc = std::move(query_mem_desc_owned);
  ctx->column_fetcher = std::move(column_fetcher);
  ctx->shared_context = std::make_unique<SharedKernelContext>(query_infos);

  ctx->co.device_type = device_type;

  return ctx;
}

ResultSetPtr Executor::runOnBatch(std::shared_ptr<StreamExecutionContext> ctx,
                                  const FragmentsList& fragments) {
  // TODO: get rid of multifragment case
  CHECK(fragments.size() == 1);
  auto query_mem_desc = *ctx->query_mem_desc;

  if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    auto metadata =
        data_mgr_->getTableMetadata(fragments[0].db_id, fragments[0].table_id);
    // TODO: think about concurrent table metadata modification

    size_t num_tuples = 0;
    for (auto f_id : fragments[0].fragment_ids) {
      auto fr = metadata->fragments[f_id];
      num_tuples = std::max(num_tuples, fr.getNumTuples());
    }

    query_mem_desc.setEntryCount(num_tuples);  // TODO(fexolm) set appropriate entry count
  }
  auto kernel = std::make_unique<ExecutionKernel>(ctx->ra_exe_unit,
                                                  ctx->co.device_type,
                                                  0,
                                                  ctx->co,
                                                  ctx->eo,
                                                  *ctx->column_fetcher,
                                                  *ctx->query_comp_desc,
                                                  query_mem_desc,
                                                  fragments,
                                                  ExecutorDispatchMode::KernelPerFragment,
                                                  -1  // TODO: rowid_lookup_key ???
  );

  kernel->run(this, 0, *ctx->shared_context);

  return nullptr;
}

hdk::ResultSetTable Executor::finishStreamExecution(
    std::shared_ptr<StreamExecutionContext> ctx) {
  for (auto& exec_ctx : ctx->shared_context->getTlsExecutionContext()) {
    if (exec_ctx) {
      CHECK(!ctx->ra_exe_unit.estimator);
      auto results =
          exec_ctx->getRowSet(ctx->ra_exe_unit, exec_ctx->query_mem_desc_, ctx->co);
      ctx->shared_context->addDeviceResults(std::move(results), 0, {});
    }
  }

  if (ctx->is_agg) {
    try {
      return collectAllDeviceResults(*ctx->shared_context,
                                     ctx->ra_exe_unit,
                                     *ctx->query_mem_desc,
                                     ctx->query_comp_desc->getDeviceType(),
                                     row_set_mem_owner_,
                                     ctx->co,
                                     ctx->eo);
    } catch (ReductionRanOutOfSlots&) {
      throw QueryExecutionError(ERR_OUT_OF_SLOTS);
    } catch (QueryExecutionError& e) {
      VLOG(1) << "Error received! error_code: " << e.getErrorCode()
              << ", what(): " << e.what();
      throw QueryExecutionError(e.getErrorCode());
    }
  }

  std::map<int, size_t> order_map;
  if (ctx->eo.preserve_order) {
    for (size_t i = 0; i < ctx->ra_exe_unit.input_descs.size(); ++i) {
      order_map[ctx->ra_exe_unit.input_descs[i].getTableId()] = i;
    }
  }
  auto result = resultsUnion(*ctx->shared_context,
                             ctx->ra_exe_unit,
                             true,  // always merge for now
                             ctx->eo.preserve_order,
                             order_map);
  return result;
}

// TODO: move this code to QueryMemoryInitializer
void Executor::allocateShuffleBuffers(
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    SharedKernelContext& shared_context) {
  CHECK(ra_exe_unit.isShuffle());
  auto partitions = ra_exe_unit.shuffle_fn->partitions;
  std::vector<TargetInfo> target_infos;
  for (auto& expr : ra_exe_unit.target_exprs) {
    CHECK(expr->is<hdk::ir::ColumnVar>()) << "Unsupported expr: " << expr->toString();
    target_infos.push_back(get_target_info(expr, getConfig().exec.group_by.bigint_count));
  }

  ColSlotContext slot_context(ra_exe_unit.target_exprs, {}, false);
  QueryMemoryDescriptor query_mem_desc(
      getDataMgr(),
      getConfigPtr(),
      query_infos,
      false /*approx_quantile*/,
      false /*topk_agg*/,
      false /*allow_multifrag*/,
      false /*keyless_hash*/,
      false /*interleaved_bins_on_gpu*/,
      -1 /*idx_target_as_key*/,
      ColRangeInfo{QueryDescriptionType::Projection, 0, 0, 0, true},
      slot_context,
      {} /*group_col_widths*/,
      8 /*group_col_compact_width*/,
      {} /*target_groupby_indices*/,
      0 /*entry_count*/,
      {} /*count_distinct_descriptors*/,
      false /*sort_on_gpu_hint*/,
      true /*output_columnar*/,
      false /*must_use_baseline_sort*/,
      false /*use_streaming_top_n*/);

  // Get buffer holding partition sizes. It is stored in the last fragment
  // of the second input table.
  CHECK_EQ(ra_exe_unit.input_descs.size(), (size_t)2);
  auto count_table_info =
      schema_provider_->getTableInfo(ra_exe_unit.input_descs.back().getTableRef());
  CHECK_EQ(count_table_info->row_count, partitions * count_table_info->fragments);
  auto count_cols = schema_provider_->listColumns(*count_table_info);
  CHECK_EQ(count_cols.size(), (size_t)2);
  auto count_col_info = count_cols.front();
  auto unpin = [](Data_Namespace::AbstractBuffer* buf) { buf->unPin(); };
  std::unique_ptr<Data_Namespace::AbstractBuffer, decltype(unpin)> sizes_buf(
      data_mgr_->getChunkBuffer({count_col_info->db_id,
                                 count_col_info->table_id,
                                 count_col_info->column_id,
                                 static_cast<int>(count_table_info->fragments)},
                                Data_Namespace::MemoryLevel::CPU_LEVEL,
                                0,
                                partitions * sizeof(uint64_t)),
      unpin);
  CHECK_EQ(sizes_buf->size(), partitions * sizeof(uint64_t));
  const uint64_t* sizes_ptr =
      reinterpret_cast<const uint64_t*>(sizes_buf->getMemoryPtr());

  // Create result set for each output partition.
  shuffle_out_bufs_.clear();
  shuffle_out_buf_ptrs_.clear();
  shuffle_out_bufs_.reserve(partitions);
  shuffle_out_buf_ptrs_.reserve(partitions);
  for (size_t i = 0; i < partitions; ++i) {
    query_mem_desc.setEntryCount(sizes_ptr[i]);
    auto rs = std::make_shared<ResultSet>(target_infos,
                                          ExecutorDeviceType::CPU,
                                          query_mem_desc,
                                          row_set_mem_owner,
                                          getDataMgr(),
                                          blockSize(),
                                          gridSize());
    rs->allocateStorage({});

    shuffle_out_bufs_.emplace_back();
    for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
         ++target_idx) {
      shuffle_out_bufs_.back().push_back(
          const_cast<int8_t*>(rs->getColumnarBuffer(target_idx)));
    }
    shuffle_out_buf_ptrs_.push_back(shuffle_out_bufs_.back().data());

    shared_context.addDeviceResults(std::move(rs), 0, {i + 1});
  }
}

std::set<ExecutorDeviceType> Executor::getAvailableDeviceTypes() const {
  std::set<ExecutorDeviceType> res{ExecutorDeviceType::CPU};
  if (data_mgr_->gpusPresent()) {
    res.insert(ExecutorDeviceType::GPU);
  }
  return res;
}

std::set<ExecutorDeviceType> Executor::getDeviceTypesForQuery(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutorDeviceType requested_dt,
    size_t& max_groups_buffer_entry_guess,
    const ExecutionOptions& eo) {
  if (needFallbackOnCPU(ra_exe_unit, requested_dt)) {
    LOG(DEBUG1) << "Devices Restricted, falling back on CPU";
    return {ExecutorDeviceType::CPU};
  }

  CHECK(!table_infos.empty());
  if (!max_groups_buffer_entry_guess) {
    LOG(DEBUG1) << "The query has failed the first execution attempt because of running "
                   "out of group by slots. Make the conservative choice: allocate "
                   "fragment size slots and run on the CPU.";
    max_groups_buffer_entry_guess = compute_buffer_entry_guess(table_infos);
    return {ExecutorDeviceType::CPU};
  }

  if (config_->exec.heterogeneous.enable_heterogeneous_execution) {
    return getAvailableDeviceTypes();
  } else {
    return {requested_dt};
  }
}

namespace {
bool has_lazy_fetched_columns(const std::vector<ColumnLazyFetchInfo>& fetched_cols) {
  for (const auto& col : fetched_cols) {
    if (col.is_lazily_fetched) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::unique_ptr<policy::ExecutionPolicy> Executor::getExecutionPolicy(
    const bool is_agg,
    const std::map<ExecutorDeviceType, std::unique_ptr<QueryMemoryDescriptor>>&
        query_mem_descs,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutionOptions& eo) {
  std::unique_ptr<policy::ExecutionPolicy> exe_policy;
  auto cfg = config_->exec.heterogeneous;

  const bool uses_lazy_fetch =
      plan_state_->allow_lazy_fetch_ &&
      has_lazy_fetched_columns(getColLazyFetchInfo(ra_exe_unit.target_exprs));

  std::map<ExecutorDeviceType, ExecutorDispatchMode> devices_dispatch_modes;

  for (const auto& dt_query_desc : query_mem_descs) {
    ExecutorDispatchMode dispatch_mode{ExecutorDispatchMode::KernelPerFragment};

    if (dt_query_desc.first == ExecutorDeviceType::GPU && eo.allow_multifrag &&
        ((!config_->exec.heterogeneous.enable_heterogeneous_execution &&
          !uses_lazy_fetch) ||
         is_agg)) {
      dispatch_mode = ExecutorDispatchMode::MultifragmentKernel;
    } else if (dt_query_desc.first == ExecutorDeviceType::CPU &&
               table_infos.size() == (size_t)1 &&
               config_->exec.group_by.enable_cpu_multifrag_kernels &&
               !ra_exe_unit.partitioned_aggregation &&
               (query_mem_descs.at(dt_query_desc.first)->getQueryDescriptionType() ==
                    QueryDescriptionType::GroupByPerfectHash ||
                query_mem_descs.at(dt_query_desc.first)->getQueryDescriptionType() ==
                    QueryDescriptionType::GroupByBaselineHash)) {
      // Right now, we don't have any heuristics to determine the perfect number of
      // kernels we want to execute on CPU and we simply create a kernel per fragment. But
      // there are some extreme cases when output buffer significanly exceeds fragment
      // size and then we have few problems:
      //  1. Output buffer initialization and reduction time exceeds fragment processing
      //     time, so it's more profitable to use a single thread for processing.
      //  2. We might simply run out of memory due to many huge hash tables and even
      //  bigger
      //     hash table created later for the reduction.
      // We need more comprehensive processing costs and memory consumption evaluation
      // here. For now, detect a simple groupby case when output hash table is bigger than
      // input table. For this case we use a single multifragment kernel to force
      // single-threaded execution with no reduction required.
      size_t input_size = table_infos.front().info->getNumTuples();
      size_t buffer_size = query_mem_descs.at(dt_query_desc.first)->getEntryCount();
      constexpr size_t threshold_ratio = 2;
      if (table_infos.front().info->fragments.size() > 1 &&
          buffer_size * threshold_ratio >= input_size) {
        LOG(INFO) << "Enabling multifrag kernels for CPU due to big output hash "
                     "table (input_size is "
                  << input_size << " rows, output buffer is " << buffer_size
                  << " entries).";
        dispatch_mode = ExecutorDispatchMode::MultifragmentKernel;
      }
    }
    devices_dispatch_modes[dt_query_desc.first] = dispatch_mode;
  }

  if (devices_dispatch_modes.size() == 1) {  // One device -> fragmentID
    exe_policy = std::make_unique<policy::FragmentIDAssignmentExecutionPolicy>(
        devices_dispatch_modes.begin()->first, devices_dispatch_modes);
  } else {
    CHECK(cfg.enable_heterogeneous_execution);

    AnalyticalTemplatesExtractor extractor;
    std::vector<costmodel::AnalyticalTemplate> templates =
        extractor.extractTemplates(ra_exe_unit);
    if (config_->exec.enable_cost_model && ra_exe_unit.cost_model != nullptr &&
        !templates.empty()) {
      size_t bytes = 0;
      // TODO(bagrorg): how can we get bytes estimation more correctly?
      for (const auto& e : table_infos) {
        auto t = e.info;
        for (const auto& f : t->fragments) {
          for (const auto& [k, v] : f.getChunkMetadataMapPhysical()) {
            bytes += v->numBytes();
          }
        }
      }
      LOG(DEBUG1) << "Cost Model enabled, making prediction for templates "
                  << toString(templates) << " for size " << bytes;

      costmodel::QueryInfo qi = {templates, bytes};
      try {
        exe_policy = ra_exe_unit.cost_model->predict(qi, devices_dispatch_modes);
        return exe_policy;
      } catch (costmodel::CostModelException& e) {
        LOG(DEBUG1) << "Cost model got an exception: " << e.what();
      }
    } else if (cfg.forced_heterogeneous_distribution) {
      std::map<ExecutorDeviceType, unsigned> distribution{
          {ExecutorDeviceType::CPU, eo.forced_cpu_proportion},
          {ExecutorDeviceType::GPU, eo.forced_gpu_proportion}};
      exe_policy = std::make_unique<policy::ProportionBasedExecutionPolicy>(
          std::move(distribution), devices_dispatch_modes);
    } else {
      exe_policy =
          std::make_unique<policy::RoundRobinExecutionPolicy>(devices_dispatch_modes);
    }
  }
  return exe_policy;
}

hdk::ResultSetTable Executor::executeWorkUnitImpl(
    size_t& max_groups_buffer_entry_guess,
    const bool is_agg,
    const bool allow_single_frag_table_opt,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool has_cardinality_estimation,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache) {
  INJECT_TIMER(Exec_executeWorkUnit);
  auto device_types_for_query = getDeviceTypesForQuery(
      ra_exe_unit, query_infos, co.device_type, max_groups_buffer_entry_guess, eo);
  CHECK_GT(device_types_for_query.size(), size_t(0));
  int8_t crt_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  do {
    SharedKernelContext shared_context(query_infos);
    ColumnFetcher column_fetcher(this, data_provider, column_cache);
    ScopeGuard scope_guard = [&column_fetcher] {
      column_fetcher.freeLinearizedBuf();
      column_fetcher.freeTemporaryCpuLinearizedIdxBuf();
    };

    if (ra_exe_unit.isShuffle()) {
      allocateShuffleBuffers(query_infos, ra_exe_unit, row_set_mem_owner, shared_context);
    }

    std::map<ExecutorDeviceType, std::unique_ptr<QueryCompilationDescriptor>>
        query_comp_descs_owned;
    std::map<ExecutorDeviceType, std::unique_ptr<QueryMemoryDescriptor>>
        query_mem_descs_owned;
    for (auto dt : device_types_for_query) {
      auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
      query_comp_desc_owned->setUseGroupByBufferDesc(co.use_groupby_buffer_desc);
      std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
      if (eo.executor_type == ExecutorType::Native) {
        try {
          INJECT_TIMER(query_step_compilation);
          query_mem_desc_owned =
              query_comp_desc_owned->compile(max_groups_buffer_entry_guess,
                                             crt_min_byte_width,
                                             has_cardinality_estimation,
                                             ra_exe_unit,
                                             query_infos,
                                             column_fetcher,
                                             {dt,
                                              co.hoist_literals,
                                              co.opt_level,
                                              co.with_dynamic_watchdog,
                                              co.allow_lazy_fetch,
                                              co.filter_on_deleted_column,
                                              co.explain_type,
                                              co.register_intel_jit_listener},
                                             eo,
                                             this);
          CHECK(query_mem_desc_owned);
          crt_min_byte_width = query_comp_desc_owned->getMinByteWidth();
        } catch (CompilationRetryNoCompaction&) {
          crt_min_byte_width = MAX_BYTE_WIDTH_SUPPORTED;
          continue;
        }
      } else {
        plan_state_.reset(new PlanState(false, query_infos, this));
        plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
        CHECK(!query_mem_desc_owned);
        query_mem_desc_owned.reset(new QueryMemoryDescriptor(
            data_mgr_, config_, 0, QueryDescriptionType::Projection, false));
      }
      const ExecutorDeviceType compiled_for_dt{query_comp_desc_owned->getDeviceType()};
      query_comp_descs_owned[compiled_for_dt] = std::move(query_comp_desc_owned);
      query_mem_descs_owned[compiled_for_dt] = std::move(query_mem_desc_owned);
    }

    const auto exe_policy =
        getExecutionPolicy(is_agg, query_mem_descs_owned, ra_exe_unit, query_infos, eo);
    const ExecutorDeviceType fallback_device{
        exe_policy->hasDevice(co.device_type) ? co.device_type : ExecutorDeviceType::CPU};
    if (eo.just_explain) {
      return {executeExplain(*query_comp_descs_owned.at(fallback_device))};
    }

    for (const auto target_expr : ra_exe_unit.target_exprs) {
      plan_state_->target_exprs_.push_back(target_expr);
    }

    if (!eo.just_validate) {
      size_t devices_count{0};
      for (const auto dt_mode : exe_policy->getExecutionModes()) {
        if (dt_mode.first == ExecutorDeviceType::CPU) {
          devices_count += cpu_threads();
        } else {
          devices_count += get_available_gpus(data_mgr_).size();
        }
      }
      CHECK_GT(devices_count, size_t(0));

      try {
        std::vector<std::unique_ptr<ExecutionKernel>> kernels;
        kernels = createKernels(shared_context,
                                ra_exe_unit,
                                column_fetcher,
                                query_infos,
                                eo,
                                co,
                                allow_single_frag_table_opt,
                                query_comp_descs_owned,
                                query_mem_descs_owned,
                                exe_policy.get(),
                                devices_count);
        launchKernels(shared_context, std::move(kernels), fallback_device, co);
      } catch (QueryExecutionError& e) {
        if (eo.with_dynamic_watchdog && interrupted_.load() &&
            e.getErrorCode() == ERR_OUT_OF_TIME) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
        if (e.getErrorCode() == ERR_INTERRUPTED) {
          throw QueryExecutionError(ERR_INTERRUPTED);
        }
        if (e.getErrorCode() == ERR_OVERFLOW_OR_UNDERFLOW &&
            static_cast<size_t>(crt_min_byte_width << 1) <= sizeof(int64_t)) {
          crt_min_byte_width <<= 1;
          continue;
        }
        throw;
      }
    }
    if (is_agg) {
      try {
        ExecutorDeviceType reduction_device_type = ExecutorDeviceType::CPU;
        if (!config_->exec.heterogeneous.enable_heterogeneous_execution) {
          reduction_device_type = fallback_device;
        }
        return collectAllDeviceResults(shared_context,
                                       ra_exe_unit,
                                       *query_mem_descs_owned[reduction_device_type],
                                       reduction_device_type,
                                       row_set_mem_owner,
                                       co,
                                       eo);
      } catch (ReductionRanOutOfSlots&) {
        throw QueryExecutionError(ERR_OUT_OF_SLOTS);
      } catch (OverflowOrUnderflow&) {
        crt_min_byte_width <<= 1;
        continue;
      } catch (QueryExecutionError& e) {
        VLOG(1) << "Error received! error_code: " << e.getErrorCode()
                << ", what(): " << e.what();
        throw QueryExecutionError(e.getErrorCode());
      }
    }
    std::map<int, size_t> order_map;
    if (eo.preserve_order) {
      for (size_t i = 0; i < ra_exe_unit.input_descs.size(); ++i) {
        order_map[ra_exe_unit.input_descs[i].getTableId()] = i;
      }
    }
    return resultsUnion(
        shared_context, ra_exe_unit, !eo.multifrag_result, eo.preserve_order, order_map);
  } while (static_cast<size_t>(crt_min_byte_width) <= sizeof(int64_t));

  return std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                     ExecutorDeviceType::CPU,
                                     QueryMemoryDescriptor(),
                                     nullptr,
                                     data_mgr_,
                                     blockSize(),
                                     gridSize());
}

void Executor::executeWorkUnitPerFragment(
    const RelAlgExecutionUnit& ra_exe_unit,
    const InputTableInfo& table_info,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    DataProvider* data_provider,
    PerFragmentCallBack& cb,
    const std::set<size_t>& fragment_indexes_param) {
  ColumnCacheMap column_cache;

  std::vector<InputTableInfo> table_infos{table_info};
  SharedKernelContext kernel_context(table_infos);

  ColumnFetcher column_fetcher(this, data_provider, column_cache);
  auto query_comp_desc_owned = std::make_unique<QueryCompilationDescriptor>();
  query_comp_desc_owned->setUseGroupByBufferDesc(co.use_groupby_buffer_desc);
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc_owned;
  {
    query_mem_desc_owned =
        query_comp_desc_owned->compile(0,
                                       8,
                                       /*has_cardinality_estimation=*/false,
                                       ra_exe_unit,
                                       table_infos,
                                       column_fetcher,
                                       co,
                                       eo,
                                       this);
  }
  CHECK(query_mem_desc_owned);
  CHECK_EQ(size_t(1), ra_exe_unit.input_descs.size());
  const auto db_id = ra_exe_unit.input_descs[0].getDatabaseId();
  const auto table_id = ra_exe_unit.input_descs[0].getTableId();
  const auto& outer_fragments = table_info.info->fragments;

  std::set<size_t> fragment_indexes;
  if (fragment_indexes_param.empty()) {
    // An empty `fragment_indexes_param` set implies executing
    // the query for all fragments in the table. In this
    // case, populate `fragment_indexes` with all fragment indexes.
    for (size_t i = 0; i < outer_fragments.size(); i++) {
      fragment_indexes.emplace(i);
    }
  } else {
    fragment_indexes = fragment_indexes_param;
  }

  {
    auto clock_begin = timer_start();
    std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
    kernel_queue_time_ms_ += timer_stop(clock_begin);

    for (auto fragment_index : fragment_indexes) {
      // We may want to consider in the future allowing this to execute on devices other
      // than CPU
      FragmentsList fragments_list{{db_id, table_id, {fragment_index}}};
      ExecutionKernel kernel(ra_exe_unit,
                             co.device_type,
                             /*device_id=*/0,
                             co,
                             eo,
                             column_fetcher,
                             *query_comp_desc_owned,
                             *query_mem_desc_owned,
                             fragments_list,
                             ExecutorDispatchMode::KernelPerFragment,
                             /*rowid_lookup_key=*/-1);
      kernel.run(this, 0, kernel_context);
    }
  }

  const auto& all_fragment_results = kernel_context.getFragmentResults();

  for (const auto& [result_set_ptr, result_fragment_indexes] : all_fragment_results) {
    CHECK_EQ(result_fragment_indexes.size(), 1);
    cb(result_set_ptr, outer_fragments[result_fragment_indexes[0]]);
  }
}

ResultSetPtr Executor::executeExplain(const QueryCompilationDescriptor& query_comp_desc) {
  return std::make_shared<ResultSet>(query_comp_desc.getIR());
}

void Executor::addTransientStringLiterals(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner) {
  auto visit_expr = [this, &row_set_mem_owner](const hdk::ir::Expr* expr) {
    if (!expr) {
      return;
    }
    const auto dict_id = TransientDictIdCollector::collect(expr);
    if (dict_id >= 0) {
      auto sdp = getStringDictionaryProxy(dict_id, row_set_mem_owner, true);
      CHECK(sdp);
      TransientStringLiteralsVisitor visitor(sdp, this);
      visitor.visit(expr);
    }
  };

  for (const auto& group_expr : ra_exe_unit.groupby_exprs) {
    visit_expr(group_expr.get());
  }

  for (const auto& group_expr : ra_exe_unit.quals) {
    visit_expr(group_expr.get());
  }

  for (const auto& quals : ra_exe_unit.join_quals) {
    for (const auto& qual_expr : quals.quals) {
      visit_expr(qual_expr.get());
    }
  }

  for (const auto& group_expr : ra_exe_unit.simple_quals) {
    visit_expr(group_expr.get());
  }

  for (const auto target_expr : ra_exe_unit.target_exprs) {
    auto target_type = target_expr->type();
    if (target_type->isString()) {
      continue;
    }
    const auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(target_expr);
    if (agg_expr) {
      if (agg_expr->aggType() == hdk::ir::AggType::kSingleValue ||
          agg_expr->aggType() == hdk::ir::AggType::kSample) {
        visit_expr(agg_expr->arg());
      }
    } else {
      visit_expr(target_expr);
    }
  }
}

ExecutorDeviceType Executor::getDeviceTypeForTargets(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType requested_device_type) {
  if (needFallbackOnCPU(ra_exe_unit, requested_device_type)) {
    return ExecutorDeviceType::CPU;
  }

  return requested_device_type;
}

bool Executor::needFallbackOnCPU(const RelAlgExecutionUnit& ra_exe_unit,
                                 const ExecutorDeviceType requested_device_type) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info =
        get_target_info(target_expr, getConfig().exec.group_by.bigint_count);
    if (!ra_exe_unit.groupby_exprs.empty() &&
        !deviceSupportsFP64(requested_device_type)) {
      if ((agg_info.agg_kind == hdk::ir::AggType::kAvg ||
           agg_info.agg_kind == hdk::ir::AggType::kSum) &&
          agg_info.agg_arg_type->isFp64()) {
        LOG(DEBUG1) << "Falling back to CPU for AVG or SUM of DOUBLE";
        return true;
      }
    }
    if (dynamic_cast<const hdk::ir::RegexpExpr*>(target_expr)) {
      LOG(DEBUG1) << "Falling back to CPU for REGEXP";
      return true;
    }
  }
  return false;
}

namespace {

int64_t inline_null_val(const hdk::ir::Type* type, const bool float_argument_input) {
  CHECK(type->isNumber() || type->isDateTime() || type->isBoolean() || type->isString() ||
        type->isExtDictionary());
  if (type->isFloatingPoint()) {
    if (float_argument_input && type->isFp32()) {
      int64_t float_null_val = 0;
      *reinterpret_cast<float*>(may_alias_ptr(&float_null_val)) =
          static_cast<float>(inline_fp_null_value(type));
      return float_null_val;
    }
    const auto double_null_val = inline_fp_null_value(type);
    return *reinterpret_cast<const int64_t*>(may_alias_ptr(&double_null_val));
  }
  return inline_int_null_value(type);
}

void fill_entries_for_empty_input(std::vector<TargetInfo>& target_infos,
                                  std::vector<int64_t>& entry,
                                  const std::vector<const hdk::ir::Expr*>& target_exprs,
                                  const QueryMemoryDescriptor& query_mem_desc,
                                  bool bigint_count) {
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    const auto agg_info = get_target_info(target_expr, bigint_count);
    CHECK(agg_info.is_agg);
    target_infos.push_back(agg_info);
    const bool float_argument_input = takes_float_argument(agg_info);
    if (agg_info.agg_kind == hdk::ir::AggType::kCount ||
        agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct) {
      entry.push_back(0);
    } else if (agg_info.agg_kind == hdk::ir::AggType::kAvg) {
      entry.push_back(0);
      entry.push_back(0);
    } else if (agg_info.agg_kind == hdk::ir::AggType::kSingleValue ||
               agg_info.agg_kind == hdk::ir::AggType::kSample) {
      if (agg_info.type->isString() || agg_info.type->isArray()) {
        entry.push_back(0);
        entry.push_back(0);
      } else {
        entry.push_back(inline_null_val(agg_info.type, float_argument_input));
      }
    } else {
      entry.push_back(inline_null_val(agg_info.type, float_argument_input));
    }
  }
}

ResultSetPtr build_row_for_empty_input(
    const Executor* executor,
    const std::vector<const hdk::ir::Expr*>& target_exprs_in,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type) {
  std::vector<hdk::ir::ExprPtr> target_exprs_owned_copies;
  std::vector<const hdk::ir::Expr*> target_exprs;
  for (const auto target_expr : target_exprs_in) {
    auto agg_expr = target_expr->as<hdk::ir::AggExpr>();
    CHECK(agg_expr);
    hdk::ir::ExprPtr target_expr_copy;
    if (agg_expr->arg()) {
      auto arg_type = agg_expr->arg()->type()->withNullable(true);
      target_expr_copy =
          hdk::ir::makeExpr<hdk::ir::AggExpr>(agg_expr->type()->withNullable(true),
                                              agg_expr->aggType(),
                                              agg_expr->arg()->withType(arg_type),
                                              agg_expr->isDistinct(),
                                              agg_expr->arg1Shared());
    } else {
      target_expr_copy = agg_expr->withType(agg_expr->type()->withNullable(true));
    }
    target_exprs_owned_copies.push_back(target_expr_copy);
    target_exprs.push_back(target_expr_copy.get());
  }
  std::vector<TargetInfo> target_infos;
  std::vector<int64_t> entry;
  fill_entries_for_empty_input(target_infos,
                               entry,
                               target_exprs,
                               query_mem_desc,
                               executor->getConfig().exec.group_by.bigint_count);
  CHECK(executor);
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto rs = std::make_shared<ResultSet>(target_infos,
                                        device_type,
                                        query_mem_desc,
                                        row_set_mem_owner,
                                        executor->getDataMgr(),
                                        executor->blockSize(),
                                        executor->gridSize());
  rs->allocateStorage();
  rs->fillOneEntry(entry);
  return rs;
}

}  // namespace

hdk::ResultSetTable Executor::collectAllDeviceResults(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  auto timer = DEBUG_TIMER(__func__);
  auto& result_per_device = shared_context.getFragmentResults();
  if (result_per_device.empty() && query_mem_desc.getQueryDescriptionType() ==
                                       QueryDescriptionType::NonGroupedAggregate) {
    return build_row_for_empty_input(
        this, ra_exe_unit.target_exprs, query_mem_desc, device_type);
  }
  if (ra_exe_unit.shuffle_fn) {
    // Reduction of shuffle COUNT(*) results.
    CHECK(ra_exe_unit.isShuffleCount()) << "unexpected shuffle results";
    return reducePartitionHistogram(result_per_device, query_mem_desc, row_set_mem_owner);
  }
  // Partitioned aggregation results don't need to be merged unless it is required
  // by execution options.
  if (ra_exe_unit.partitioned_aggregation && eo.multifrag_result) {
    return get_separate_results(result_per_device);
  }
  if (use_speculative_top_n(ra_exe_unit, query_mem_desc)) {
    try {
      return reduceSpeculativeTopN(
          ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc);
    } catch (const std::bad_alloc&) {
      throw SpeculativeTopNFailed("Failed during multi-device reduction.");
    }
  }
  return reduceMultiDeviceResults(
      ra_exe_unit, result_per_device, row_set_mem_owner, query_mem_desc, co);
}

std::unordered_map<int, const hdk::ir::BinOper*> Executor::getInnerTabIdToJoinCond()
    const {
  std::unordered_map<int, const hdk::ir::BinOper*> id_to_cond;
  const auto& join_info = plan_state_->join_info_;
  CHECK_EQ(join_info.equi_join_tautologies_.size(), join_info.join_hash_tables_.size());
  for (size_t i = 0; i < join_info.join_hash_tables_.size(); ++i) {
    int inner_table_id = join_info.join_hash_tables_[i]->getInnerTableId();
    id_to_cond.insert(
        std::make_pair(inner_table_id, join_info.equi_join_tautologies_[i].get()));
  }
  return id_to_cond;
}

std::vector<std::unique_ptr<ExecutionKernel>> Executor::createKernels(
    SharedKernelContext& shared_context,
    const RelAlgExecutionUnit& ra_exe_unit,
    ColumnFetcher& column_fetcher,
    const std::vector<InputTableInfo>& table_infos,
    const ExecutionOptions& eo,
    const CompilationOptions& co,
    const bool allow_single_frag_table_opt,
    const std::map<ExecutorDeviceType, std::unique_ptr<QueryCompilationDescriptor>>&
        query_comp_descs,
    const std::map<ExecutorDeviceType, std::unique_ptr<QueryMemoryDescriptor>>&
        query_mem_descs,
    const policy::ExecutionPolicy* policy,
    const size_t device_count) {
  std::vector<std::unique_ptr<ExecutionKernel>> execution_kernels;

  QueryFragmentDescriptor fragment_descriptor(
      ra_exe_unit,
      table_infos,
      data_mgr_->getMemoryInfo(Data_Namespace::MemoryLevel::GPU_LEVEL),
      eo.gpu_input_mem_limit_percent,
      eo.outer_fragment_indices);
  CHECK(!ra_exe_unit.input_descs.empty());

  fragment_descriptor.buildFragmentKernelMap(
      ra_exe_unit, shared_context.getFragOffsets(), policy, this, co.codegen_traits_desc);

  if (!config_->exec.heterogeneous.enable_heterogeneous_execution && eo.with_watchdog &&
      fragment_descriptor.shouldCheckWorkUnitWatchdog()) {
    checkWorkUnitWatchdog(ra_exe_unit,
                          table_infos,
                          *schema_provider_,
                          query_comp_descs.begin()->first,
                          device_count);
  }

  for (const auto& dt_query_desc : query_mem_descs) {
    if (policy->getExecutionMode(dt_query_desc.first) ==
        ExecutorDispatchMode::KernelPerFragment) {
      VLOG(1) << "Dispatching one execution kernel per fragment";
      VLOG(1) << dt_query_desc.second->toString();
      if (allow_single_frag_table_opt &&
          (dt_query_desc.second->getQueryDescriptionType() ==
           QueryDescriptionType::Projection) &&
          table_infos.size() == 1) {
        const auto max_frag_size =
            table_infos.front().info->getFragmentNumTuplesUpperBound();
        if (max_frag_size < dt_query_desc.second->getEntryCount()) {
          LOG(INFO) << "Lowering scan limit from "
                    << dt_query_desc.second->getEntryCount()
                    << " to match max fragment size " << max_frag_size
                    << " for kernel per fragment execution path.";
          throw CompilationRetryNewScanLimit(max_frag_size);
        }
      }
    }
  }

  size_t frag_list_idx{0};
  auto kernel_dispatch = [&ra_exe_unit,
                          &execution_kernels,
                          &column_fetcher,
                          &co,
                          &eo,
                          &frag_list_idx,
                          &query_comp_descs,
                          &query_mem_descs,
                          policy](const int device_id,
                                  const FragmentsList& frag_list,
                                  const int64_t rowid_lookup_key,
                                  const ExecutorDeviceType device_type) {
    if (!frag_list.size()) {
      return;
    }
    CHECK_GE(device_id, 0);
    CHECK(query_comp_descs.count(device_type));
    CHECK(query_mem_descs.count(device_type));
    execution_kernels.emplace_back(
        std::make_unique<ExecutionKernel>(ra_exe_unit,
                                          device_type,
                                          device_id,
                                          co,
                                          eo,
                                          column_fetcher,
                                          *query_comp_descs.at(device_type).get(),
                                          *query_mem_descs.at(device_type).get(),
                                          frag_list,
                                          policy->getExecutionMode(device_type),
                                          rowid_lookup_key));

    ++frag_list_idx;
  };
  fragment_descriptor.dispatchKernelsToDevices(kernel_dispatch, ra_exe_unit, policy);
  return execution_kernels;
}

// TODO(Petr): remove device_type from function signature
void Executor::launchKernels(SharedKernelContext& shared_context,
                             std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
                             const ExecutorDeviceType device_type,
                             const CompilationOptions& co) {
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
  kernel_queue_time_ms_ += timer_stop(clock_begin);

  tbb::task_group tg;
  // A hack to have unused unit for results collection.
  const RelAlgExecutionUnit* ra_exe_unit =
      kernels.empty() ? nullptr : &kernels[0]->ra_exe_unit_;

  if (config_->exec.sub_tasks.enable && device_type == ExecutorDeviceType::CPU) {
    shared_context.setThreadPool(&tg);
  }
  ScopeGuard pool_guard([&shared_context]() { shared_context.setThreadPool(nullptr); });

  VLOG(1) << "Launching " << kernels.size() << " kernels for query on: ";
  for (size_t i = 0; i < kernels.size(); i++) {
    VLOG(1) << "\t" << i << ' ' << (toString(kernels[i])) << ".";
  }

  size_t kernel_idx = 1;
  for (auto& kernel : kernels) {
    CHECK(kernel.get());
    tg.run([this,
            &kernel,
            &shared_context,
            parent_thread_id = logger::thread_id(),
            crt_kernel_idx = kernel_idx++] {
      DEBUG_TIMER_NEW_THREAD(parent_thread_id);
      const size_t thread_i = crt_kernel_idx % cpu_threads();
      kernel->run(this, thread_i, shared_context);
    });
  }
  tg.wait();

  for (auto& exec_ctx : shared_context.getTlsExecutionContext()) {
    // The first arg is used for GPU only, it's not our case.
    // TODO: add QueryExecutionContext::getRowSet() interface
    // for our case.
    if (exec_ctx) {
      ResultSetPtr results;
      if (ra_exe_unit->estimator) {
        results = std::shared_ptr<ResultSet>(exec_ctx->estimator_result_set_.release());
      } else {
        results = exec_ctx->getRowSet(*ra_exe_unit, exec_ctx->query_mem_desc_, co);
      }
      shared_context.addDeviceResults(std::move(results), 0, {});
    }
  }
}

std::vector<size_t> Executor::getTableFragmentIndices(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const size_t table_idx,
    const size_t outer_frag_idx,
    std::map<TableRef, const TableFragments*>& selected_tables_fragments,
    const std::unordered_map<int, const hdk::ir::BinOper*>&
        inner_table_id_to_join_condition) {
  const auto table_ref = ra_exe_unit.input_descs[table_idx].getTableRef();
  auto table_frags_it = selected_tables_fragments.find(table_ref);
  CHECK(table_frags_it != selected_tables_fragments.end());
  const auto& outer_input_desc = ra_exe_unit.input_descs[0];
  const auto outer_table_fragments_it =
      selected_tables_fragments.find(outer_input_desc.getTableRef());
  const auto outer_table_fragments = outer_table_fragments_it->second;
  CHECK(outer_table_fragments_it != selected_tables_fragments.end());
  CHECK_LT(outer_frag_idx, outer_table_fragments->size());
  if (!table_idx || ra_exe_unit.isShuffle()) {
    return {outer_frag_idx};
  }
  const auto& outer_fragment_info = (*outer_table_fragments)[outer_frag_idx];
  auto& inner_frags = table_frags_it->second;
  CHECK_LT(size_t(1), ra_exe_unit.input_descs.size());
  std::vector<size_t> all_frag_ids;
  for (size_t inner_frag_idx = 0; inner_frag_idx < inner_frags->size();
       ++inner_frag_idx) {
    const auto& inner_frag_info = (*inner_frags)[inner_frag_idx];
    if (skipFragmentPair(outer_fragment_info,
                         inner_frag_info,
                         table_idx,
                         inner_table_id_to_join_condition,
                         ra_exe_unit,
                         device_type)) {
      continue;
    }
    all_frag_ids.push_back(inner_frag_idx);
  }
  return all_frag_ids;
}

// Returns true iff the join between two fragments cannot yield any results, per
// shard information. The pair can be skipped to avoid full broadcast.
bool Executor::skipFragmentPair(const FragmentInfo& outer_fragment_info,
                                const FragmentInfo& inner_fragment_info,
                                const int table_idx,
                                const std::unordered_map<int, const hdk::ir::BinOper*>&
                                    inner_table_id_to_join_condition,
                                const RelAlgExecutionUnit& ra_exe_unit,
                                const ExecutorDeviceType device_type) {
  return false;
}

std::map<size_t, std::vector<uint64_t>> get_table_id_to_frag_offsets(
    const std::vector<InputDescriptor>& input_descs,
    const std::map<TableRef, const TableFragments*>& all_tables_fragments) {
  std::map<size_t, std::vector<uint64_t>> tab_id_to_frag_offsets;
  for (auto& desc : input_descs) {
    const auto fragments_it = all_tables_fragments.find(desc.getTableRef());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto& fragments = *fragments_it->second;
    std::vector<uint64_t> frag_offsets(fragments.size(), 0);
    for (size_t i = 0, off = 0; i < fragments.size(); ++i) {
      frag_offsets[i] = off;
      off += fragments[i].getNumTuples();
    }
    tab_id_to_frag_offsets.insert(std::make_pair(desc.getTableId(), frag_offsets));
  }
  return tab_id_to_frag_offsets;
}

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
Executor::getRowCountAndOffsetForAllFrags(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<std::vector<size_t>>& frag_ids_crossjoin,
    const std::vector<InputDescriptor>& input_descs,
    const std::map<TableRef, const TableFragments*>& all_tables_fragments) {
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  const auto tab_id_to_frag_offsets =
      get_table_id_to_frag_offsets(input_descs, all_tables_fragments);
  std::unordered_map<size_t, size_t> outer_id_to_num_row_idx;
  for (const auto& selected_frag_ids : frag_ids_crossjoin) {
    std::vector<int64_t> num_rows;
    std::vector<uint64_t> frag_offsets;
    if (!ra_exe_unit.union_all) {
      CHECK_EQ(selected_frag_ids.size(), input_descs.size());
    }
    for (size_t tab_idx = 0; tab_idx < input_descs.size(); ++tab_idx) {
      const auto frag_id = ra_exe_unit.union_all ? 0 : selected_frag_ids[tab_idx];
      const auto fragments_it =
          all_tables_fragments.find(input_descs[tab_idx].getTableRef());
      CHECK(fragments_it != all_tables_fragments.end());
      const auto& fragments = *fragments_it->second;
      if (ra_exe_unit.join_quals.empty() || tab_idx == 0 ||
          plan_state_->join_info_.sharded_range_table_indices_.count(tab_idx)) {
        const auto& fragment = fragments[frag_id];
        num_rows.push_back(fragment.getNumTuples());
      } else {
        size_t total_row_count{0};
        for (const auto& fragment : fragments) {
          total_row_count += fragment.getNumTuples();
        }
        num_rows.push_back(total_row_count);
      }
      const auto frag_offsets_it =
          tab_id_to_frag_offsets.find(input_descs[tab_idx].getTableId());
      CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
      const auto& offsets = frag_offsets_it->second;
      CHECK_LT(frag_id, offsets.size());
      frag_offsets.push_back(offsets[frag_id]);
    }
    all_num_rows.push_back(num_rows);
    // Fragment offsets of outer table should be ONLY used by rowid for now.
    all_frag_offsets.push_back(frag_offsets);
  }
  return {all_num_rows, all_frag_offsets};
}

// Only fetch columns of hash-joined inner fact table whose fetch are not deferred from
// all the table fragments.
bool Executor::needFetchAllFragments(const InputColDescriptor& inner_col_desc,
                                     const RelAlgExecutionUnit& ra_exe_unit,
                                     const FragmentsList& selected_fragments) const {
  const auto& input_descs = ra_exe_unit.input_descs;
  const int nest_level = inner_col_desc.getNestLevel();
  if (nest_level < 1 || ra_exe_unit.join_quals.empty() || input_descs.size() < 2 ||
      (ra_exe_unit.join_quals.empty() &&
       plan_state_->isLazyFetchColumn(inner_col_desc))) {
    return false;
  }
  const int table_id = inner_col_desc.getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].table_id);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  return fragments.size() > 1;
}

bool Executor::needLinearizeAllFragments(
    const InputColDescriptor& inner_col_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentsList& selected_fragments,
    const Data_Namespace::MemoryLevel memory_level) const {
  const int nest_level = inner_col_desc.getNestLevel();
  const int table_id = inner_col_desc.getTableId();
  CHECK_LT(static_cast<size_t>(nest_level), selected_fragments.size());
  CHECK_EQ(table_id, selected_fragments[nest_level].table_id);
  const auto& fragments = selected_fragments[nest_level].fragment_ids;
  auto need_linearize =
      inner_col_desc.type()->isArray() || inner_col_desc.type()->isString();
  return need_linearize && fragments.size() > 1;
}

std::ostream& operator<<(std::ostream& os, FetchResult const& fetch_result) {
  return os << "col_buffers" << shared::printContainer(fetch_result.col_buffers)
            << " num_rows" << shared::printContainer(fetch_result.num_rows)
            << " frag_offsets" << shared::printContainer(fetch_result.frag_offsets);
}

FetchResult Executor::fetchChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<TableRef, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchChunks);
  const auto& col_global_ids = ra_exe_unit.input_col_descs;
  std::vector<std::vector<size_t>> selected_fragments_crossjoin;
  std::vector<size_t> local_col_to_frag_pos;
  buildSelectedFragsMapping(selected_fragments_crossjoin,
                            local_col_to_frag_pos,
                            col_global_ids,
                            selected_fragments,
                            ra_exe_unit);

  CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
      selected_fragments_crossjoin);
  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;

  auto fetch_column_callback = [&](std::shared_ptr<const InputColDescriptor> col_id,
                                   const std::vector<size_t>& selected_frag_ids,
                                   std::vector<const int8_t*>& frag_col_buffers,
                                   const bool parallelized =
                                       false) -> bool /*empty_frag*/ {
    if (interrupted_.load()) {
      throw QueryExecutionError(ERR_INTERRUPTED);
    }
    const auto fragments_it = all_tables_fragments.find(col_id->getTableRef());
    CHECK(fragments_it != all_tables_fragments.end());
    const auto fragments = fragments_it->second;
    auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
    CHECK(it != plan_state_->global_to_local_col_ids_.end());
    CHECK_LT(static_cast<size_t>(it->second),
             plan_state_->global_to_local_col_ids_.size());
    const size_t frag_id = selected_frag_ids[local_col_to_frag_pos[it->second]];
    if (!fragments->size()) {
      return true;
    }
    auto memory_level_for_column = memory_level;
    if (plan_state_->columns_to_fetch_.find(*col_id) ==
        plan_state_->columns_to_fetch_.end()) {
      memory_level_for_column = Data_Namespace::CPU_LEVEL;
    }
    if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
      // determine if we need special treatment to linearlize multi-frag table
      // i.e., a column that is classified as varlen type, i.e., array
      // for now, we can support more types in this way

      // If fetch_column_callback is called from parallel code region, we get deadlocks
      // due to tbb nested parallel_for, the calls below are internally parallelized. We
      // should land here when called from a sequential code region
      CHECK(!parallelized);
      if (needLinearizeAllFragments(
              *col_id, ra_exe_unit, selected_fragments, memory_level)) {
        bool for_lazy_fetch = false;
        if (plan_state_->columns_to_not_fetch_.find(*col_id) !=
            plan_state_->columns_to_not_fetch_.end()) {
          for_lazy_fetch = true;
          VLOG(2) << "Try to linearize lazy fetch column (col_id: " << col_id->getColId()
                  << ")";
        }
        frag_col_buffers[it->second] = column_fetcher.linearizeColumnFragments(
            col_id->getColInfo(),
            all_tables_fragments,
            chunks,
            chunk_iterators,
            for_lazy_fetch ? Data_Namespace::CPU_LEVEL : memory_level,
            for_lazy_fetch ? 0 : device_id,
            device_allocator);
      } else {
        frag_col_buffers[it->second] =
            column_fetcher.getAllTableColumnFragments(col_id->getColInfo(),
                                                      all_tables_fragments,
                                                      memory_level_for_column,
                                                      device_id,
                                                      device_allocator,
                                                      thread_idx);
      }
    } else {
      frag_col_buffers[it->second] =
          column_fetcher.getOneTableColumnFragment(col_id->getColInfo(),
                                                   frag_id,
                                                   all_tables_fragments,
                                                   chunks,
                                                   chunk_iterators,
                                                   memory_level_for_column,
                                                   device_id,
                                                   device_allocator);
    }
    return false;
  };

  // in MT fetching for GPU, we want to preserve "the order of insertion" into
  // all_frag_col_buffers
  std::vector<std::vector<size_t>> selected_frag_ids_vec;
  if (memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL) {
    std::atomic<bool> empty_frags{false};
    tbb::task_arena limitedArena(8);
    std::vector<size_t> idx_frags_to_inearize;
    for (const auto& selected_frag_ids : frag_ids_crossjoin) {
      selected_frag_ids_vec.push_back(selected_frag_ids);
      for (const auto& col_id : col_global_ids) {
        CHECK(col_id);
        if (!col_id->isVirtual() &&
            needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          idx_frags_to_inearize.push_back(selected_frag_ids_vec.size() - 1);
        }
      }
    }
    all_frag_col_buffers.resize(selected_frag_ids_vec.size());

    // Try MT fetching for frags that do not need linearization
    limitedArena.execute([&]() {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, selected_frag_ids_vec.size()), [&](auto r) {
            for (size_t idx = r.begin(); idx != r.end(); ++idx) {
              if (std::find(idx_frags_to_inearize.begin(),
                            idx_frags_to_inearize.end(),
                            idx) == idx_frags_to_inearize.end()) {
                const auto& selected_frag_ids = selected_frag_ids_vec[idx];
                std::vector<const int8_t*> frag_col_buffers(
                    plan_state_->global_to_local_col_ids_.size());
                for (const auto& col_id : col_global_ids) {
                  CHECK(col_id);
                  if (!col_id->isVirtual() &&
                      fetch_column_callback(
                          col_id, selected_frag_ids, frag_col_buffers, true)) {
                    empty_frags = true;  // not virtual, but empty frags
                    tbb::task::current_context()->cancel_group_execution();
                  }
                }
                all_frag_col_buffers[idx] = frag_col_buffers;
              }
            }
          });
    });
    if (empty_frags) {
      return {};
    }
    for (const size_t idx :
         idx_frags_to_inearize) {  // linear frags materialization is already
                                   // parallelized, avoid nested tbb
      const auto& selected_frag_ids = selected_frag_ids_vec[idx];
      std::vector<const int8_t*> frag_col_buffers(
          plan_state_->global_to_local_col_ids_.size());
      for (const auto& col_id : col_global_ids) {
        CHECK(col_id);
        if (!col_id->isVirtual() &&
            fetch_column_callback(col_id, selected_frag_ids, frag_col_buffers)) {
          return {};  // not virtual, but empty frags
        }
      }
      all_frag_col_buffers[idx] = frag_col_buffers;
    }
  } else {
    for (const auto& selected_frag_ids : frag_ids_crossjoin) {
      std::vector<const int8_t*> frag_col_buffers(
          plan_state_->global_to_local_col_ids_.size());
      for (const auto& col_id : col_global_ids) {
        CHECK(col_id);
        if (!col_id->isVirtual() &&
            fetch_column_callback(col_id, selected_frag_ids, frag_col_buffers)) {
          return {};  // not virtual, but empty frags
        }
      }
      selected_frag_ids_vec.push_back(selected_frag_ids);
      all_frag_col_buffers.push_back(frag_col_buffers);
    }
  }
  // selected_frag_ids_vec here is just a vector representation of frag_ids_crossjoin
  // we need it in this form to have a proper fragment order when doing MT fetching to GPU
  std::tie(all_num_rows, all_frag_offsets) = getRowCountAndOffsetForAllFrags(
      ra_exe_unit, selected_frag_ids_vec, ra_exe_unit.input_descs, all_tables_fragments);
  return {all_frag_col_buffers, all_num_rows, all_frag_offsets};
}

// fetchChunks() is written under the assumption that multiple inputs implies a JOIN.
// This is written under the assumption that multiple inputs implies a UNION ALL.
FetchResult Executor::fetchUnionChunks(
    const ColumnFetcher& column_fetcher,
    const RelAlgExecutionUnit& ra_exe_unit,
    const int device_id,
    const Data_Namespace::MemoryLevel memory_level,
    const std::map<TableRef, const TableFragments*>& all_tables_fragments,
    const FragmentsList& selected_fragments,
    std::list<ChunkIter>& chunk_iterators,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const bool allow_runtime_interrupt) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(fetchUnionChunks);

  std::vector<std::vector<const int8_t*>> all_frag_col_buffers;
  std::vector<std::vector<int64_t>> all_num_rows;
  std::vector<std::vector<uint64_t>> all_frag_offsets;
  std::vector<std::vector<size_t>> selected_frag_ids_vec;

  CHECK(!selected_fragments.empty());
  CHECK_LE(2u, ra_exe_unit.input_descs.size());
  CHECK_LE(2u, ra_exe_unit.input_col_descs.size());
  using TableId = int;
  TableId const selected_table_id = selected_fragments.front().table_id;
  bool const input_descs_index =
      selected_table_id == ra_exe_unit.input_descs[1].getTableId();
  if (!input_descs_index) {
    CHECK_EQ(selected_table_id, ra_exe_unit.input_descs[0].getTableId());
  }
  bool const input_col_descs_index =
      selected_table_id ==
      (*std::next(ra_exe_unit.input_col_descs.begin()))->getTableId();
  if (!input_col_descs_index) {
    CHECK_EQ(selected_table_id, ra_exe_unit.input_col_descs.front()->getTableId());
  }
  VLOG(2) << "selected_fragments.size()=" << selected_fragments.size()
          << " selected_table_id=" << selected_table_id
          << " input_descs_index=" << int(input_descs_index)
          << " input_col_descs_index=" << int(input_col_descs_index)
          << " ra_exe_unit.input_descs="
          << shared::printContainer(ra_exe_unit.input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs);

  // Partition col_global_ids by table_id
  std::unordered_map<TableId, std::list<std::shared_ptr<const InputColDescriptor>>>
      table_id_to_input_col_descs;
  for (auto const& input_col_desc : ra_exe_unit.input_col_descs) {
    TableId const table_id = input_col_desc->getTableId();
    table_id_to_input_col_descs[table_id].push_back(input_col_desc);
  }
  for (auto const& pair : table_id_to_input_col_descs) {
    std::vector<std::vector<size_t>> selected_fragments_crossjoin;
    std::vector<size_t> local_col_to_frag_pos;

    buildSelectedFragsMappingForUnion(selected_fragments_crossjoin,
                                      local_col_to_frag_pos,
                                      pair.second,
                                      selected_fragments,
                                      ra_exe_unit);

    CartesianProduct<std::vector<std::vector<size_t>>> frag_ids_crossjoin(
        selected_fragments_crossjoin);

    for (const auto& selected_frag_ids : frag_ids_crossjoin) {
      if (interrupted_.load()) {
        throw QueryExecutionError(ERR_INTERRUPTED);
      }
      std::vector<const int8_t*> frag_col_buffers(
          plan_state_->global_to_local_col_ids_.size());
      for (const auto& col_id : pair.second) {
        CHECK(col_id);
        const int table_id = col_id->getTableId();
        CHECK_EQ(table_id, pair.first);
        if (col_id->isVirtual()) {
          continue;
        }
        const auto fragments_it = all_tables_fragments.find(col_id->getTableRef());
        CHECK(fragments_it != all_tables_fragments.end());
        const auto fragments = fragments_it->second;
        auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
        CHECK(it != plan_state_->global_to_local_col_ids_.end());
        CHECK_LT(static_cast<size_t>(it->second),
                 plan_state_->global_to_local_col_ids_.size());
        const size_t frag_id = ra_exe_unit.union_all
                                   ? 0
                                   : selected_frag_ids[local_col_to_frag_pos[it->second]];
        if (!fragments->size()) {
          return {};
        }
        CHECK_LT(frag_id, fragments->size());
        auto memory_level_for_column = memory_level;
        if (plan_state_->columns_to_fetch_.find(*col_id) ==
            plan_state_->columns_to_fetch_.end()) {
          memory_level_for_column = Data_Namespace::CPU_LEVEL;
        }
        if (needFetchAllFragments(*col_id, ra_exe_unit, selected_fragments)) {
          frag_col_buffers[it->second] =
              column_fetcher.getAllTableColumnFragments(col_id->getColInfo(),
                                                        all_tables_fragments,
                                                        memory_level_for_column,
                                                        device_id,
                                                        device_allocator,
                                                        thread_idx);
        } else {
          frag_col_buffers[it->second] =
              column_fetcher.getOneTableColumnFragment(col_id->getColInfo(),
                                                       frag_id,
                                                       all_tables_fragments,
                                                       chunks,
                                                       chunk_iterators,
                                                       memory_level_for_column,
                                                       device_id,
                                                       device_allocator);
        }
      }
      selected_frag_ids_vec.push_back(selected_frag_ids);
      all_frag_col_buffers.push_back(frag_col_buffers);
    }
    std::vector<std::vector<int64_t>> num_rows;
    std::vector<std::vector<uint64_t>> frag_offsets;
    std::tie(num_rows, frag_offsets) =
        getRowCountAndOffsetForAllFrags(ra_exe_unit,
                                        selected_frag_ids_vec,
                                        ra_exe_unit.input_descs,
                                        all_tables_fragments);
    all_num_rows.insert(all_num_rows.end(), num_rows.begin(), num_rows.end());
    all_frag_offsets.insert(
        all_frag_offsets.end(), frag_offsets.begin(), frag_offsets.end());
  }
  // The hack below assumes a particular table traversal order which is not
  // always achieved due to unordered map in the outermost loop. According
  // to the code below we expect NULLs in even positions of all_frag_col_buffers[0]
  // and odd positions of all_frag_col_buffers[1]. As an additional hack we
  // swap these vectors if NULLs are not on expected positions.
  if (all_frag_col_buffers[0].size() > 1 && all_frag_col_buffers[0][0] &&
      !all_frag_col_buffers[0][1]) {
    std::swap(all_frag_col_buffers[0], all_frag_col_buffers[1]);
  }
  // UNION ALL hacks.
  VLOG(2) << "all_frag_col_buffers=" << shared::printContainer(all_frag_col_buffers);
  for (size_t i = 0; i < all_frag_col_buffers.front().size(); ++i) {
    all_frag_col_buffers[i & 1][i] = all_frag_col_buffers[i & 1][i ^ 1];
  }
  if (input_descs_index == input_col_descs_index) {
    std::swap(all_frag_col_buffers[0], all_frag_col_buffers[1]);
  }

  VLOG(2) << "all_frag_col_buffers=" << shared::printContainer(all_frag_col_buffers)
          << " all_num_rows=" << shared::printContainer(all_num_rows)
          << " all_frag_offsets=" << shared::printContainer(all_frag_offsets)
          << " input_col_descs_index=" << input_col_descs_index;
  return {{all_frag_col_buffers[input_descs_index]},
          {{all_num_rows[0][input_descs_index]}},
          {{all_frag_offsets[0][input_descs_index]}}};
}

std::vector<size_t> Executor::getFragmentCount(const FragmentsList& selected_fragments,
                                               const size_t scan_idx,
                                               const RelAlgExecutionUnit& ra_exe_unit) {
  if ((ra_exe_unit.input_descs.size() > size_t(2) || !ra_exe_unit.join_quals.empty()) &&
      scan_idx > 0 &&
      !plan_state_->join_info_.sharded_range_table_indices_.count(scan_idx) &&
      !selected_fragments[scan_idx].fragment_ids.empty()) {
    // Fetch all fragments
    return {size_t(0)};
  }

  return selected_fragments[scan_idx].fragment_ids;
}

void Executor::buildSelectedFragsMapping(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    std::vector<size_t>& local_col_to_frag_pos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    CHECK_EQ(selected_fragments[scan_idx].table_id, table_id);
    selected_fragments_crossjoin.push_back(
        getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      if (col_id->getTableId() != table_id ||
          col_id->getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

void Executor::buildSelectedFragsMappingForUnion(
    std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
    std::vector<size_t>& local_col_to_frag_pos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
    const FragmentsList& selected_fragments,
    const RelAlgExecutionUnit& ra_exe_unit) {
  local_col_to_frag_pos.resize(plan_state_->global_to_local_col_ids_.size());
  size_t frag_pos{0};
  const auto& input_descs = ra_exe_unit.input_descs;
  for (size_t scan_idx = 0; scan_idx < input_descs.size(); ++scan_idx) {
    const int table_id = input_descs[scan_idx].getTableId();
    // selected_fragments here is from assignFragsToKernelDispatch
    // execution_kernel.fragments
    if (selected_fragments[0].table_id != table_id) {  // TODO 0
      continue;
    }
    // CHECK_EQ(selected_fragments[scan_idx].table_id, table_id);
    selected_fragments_crossjoin.push_back(
        // getFragmentCount(selected_fragments, scan_idx, ra_exe_unit));
        {size_t(1)});  // TODO
    for (const auto& col_id : col_global_ids) {
      CHECK(col_id);
      if (col_id->getTableId() != table_id ||
          col_id->getNestLevel() != static_cast<int>(scan_idx)) {
        continue;
      }
      auto it = plan_state_->global_to_local_col_ids_.find(*col_id);
      CHECK(it != plan_state_->global_to_local_col_ids_.end());
      CHECK_LT(static_cast<size_t>(it->second),
               plan_state_->global_to_local_col_ids_.size());
      local_col_to_frag_pos[it->second] = frag_pos;
    }
    ++frag_pos;
  }
}

namespace {

class OutVecOwner {
 public:
  OutVecOwner(const std::vector<int64_t*>& out_vec) : out_vec_(out_vec) {}
  ~OutVecOwner() {
    for (auto out : out_vec_) {
      delete[] out;
    }
  }

 private:
  std::vector<int64_t*> out_vec_;
};

bool check_rows_less_than_needed(const ResultSetPtr& results, const size_t scan_limit) {
  CHECK(scan_limit);
  return results && results->rowCount() < scan_limit;
}

}  // namespace

int32_t Executor::executePlan(const RelAlgExecutionUnit& ra_exe_unit,
                              const CompilationResult& compilation_result,
                              const bool hoist_literals,
                              ResultSetPtr* results,
                              const ExecutorDeviceType device_type,
                              const CompilationOptions& co,
                              std::vector<std::vector<const int8_t*>>& col_buffers,
                              const std::vector<size_t> outer_tab_frag_ids,
                              QueryExecutionContext* query_exe_context,
                              const std::vector<std::vector<int64_t>>& num_rows,
                              const std::vector<std::vector<uint64_t>>& frag_offsets,
                              Data_Namespace::DataMgr* data_mgr,
                              const int device_id,
                              const int outer_table_id,
                              const int64_t scan_limit,
                              const uint32_t start_rowid,
                              const uint32_t num_tables,
                              const bool allow_runtime_interrupt,
                              const int64_t rows_to_process) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executePlan);
  // TODO: get results via a separate method, but need to do something with literals.
  CHECK(!results || !(*results));
  if (col_buffers.empty()) {
    return 0;
  }

  const bool is_groupby = ra_exe_unit.groupby_exprs.size() != 0;
  // TODO(alex):
  // 1. Optimize size (make keys more compact).
  // 2. Resize on overflow.
  // 3. Optimize runtime.
  auto hoist_buf = serializeLiterals(compilation_result.literal_values, device_id);
  int32_t error_code = device_type == ExecutorDeviceType::GPU ? 0 : start_rowid;
  const auto join_hash_table_ptrs = getJoinHashTablePtrs(device_type, device_id);
  if (interrupted_.load()) {
    throw QueryExecutionError(ERR_INTERRUPTED);
  }

  VLOG(2) << "bool(ra_exe_unit.union_all)=" << bool(ra_exe_unit.union_all)
          << " ra_exe_unit.input_descs="
          << shared::printContainer(ra_exe_unit.input_descs)
          << " ra_exe_unit.input_col_descs="
          << shared::printContainer(ra_exe_unit.input_col_descs)
          << " ra_exe_unit.scan_limit=" << ra_exe_unit.scan_limit
          << " num_rows=" << shared::printContainer(num_rows)
          << " frag_offsets=" << shared::printContainer(frag_offsets)
          << " query_exe_context->query_buffers_->num_rows_="
          << query_exe_context->query_buffers_->num_rows_
          << " query_exe_context->query_mem_desc_.getEntryCount()="
          << query_exe_context->query_mem_desc_.getEntryCount()
          << " device_id=" << device_id << " outer_table_id=" << outer_table_id
          << " scan_limit=" << scan_limit << " start_rowid=" << start_rowid
          << " num_tables=" << num_tables;

  if (!is_groupby) {
    std::unique_ptr<OutVecOwner> output_memory_scope;
    std::vector<int64_t*> out_vec;
    if (device_type == ExecutorDeviceType::CPU) {
      CpuCompilationContext* cpu_generated_code =
          dynamic_cast<CpuCompilationContext*>(compilation_result.generated_code.get());
      CHECK(cpu_generated_code);
      out_vec = query_exe_context->launchCpuCode(ra_exe_unit,
                                                 cpu_generated_code,
                                                 hoist_literals,
                                                 hoist_buf,
                                                 col_buffers,
                                                 num_rows,
                                                 frag_offsets,
                                                 0,
                                                 &error_code,
                                                 num_tables,
                                                 join_hash_table_ptrs,
                                                 rows_to_process);
      output_memory_scope.reset(new OutVecOwner(out_vec));
    } else {
      CompilationContext* gpu_generated_code = compilation_result.generated_code.get();
      CHECK(gpu_generated_code);
      try {
        out_vec = query_exe_context->launchGpuCode(
            ra_exe_unit,
            gpu_generated_code,
            hoist_literals,
            hoist_buf,
            col_buffers,
            num_rows,
            frag_offsets,
            0,
            data_mgr,
            getBufferProvider(),
            blockSize(),
            gridSize(),
            device_id,
            compilation_result.gpu_smem_context.getSharedMemorySize(),
            &error_code,
            num_tables,
            allow_runtime_interrupt,
            join_hash_table_ptrs);
        output_memory_scope.reset(new OutVecOwner(out_vec));
      } catch (const OutOfMemory&) {
        return ERR_OUT_OF_GPU_MEM;
      } catch (const std::exception& e) {
        LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
      }
    }
    if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
        error_code == Executor::ERR_DIV_BY_ZERO ||
        error_code == Executor::ERR_OUT_OF_TIME ||
        error_code == Executor::ERR_INTERRUPTED ||
        error_code == Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES ||
        error_code == Executor::ERR_WIDTH_BUCKET_INVALID_ARGUMENT) {
      return error_code;
    }
    if (ra_exe_unit.estimator) {
      CHECK(!error_code);
      if (results) {
        *results = std::shared_ptr<ResultSet>(
            query_exe_context->estimator_result_set_.release());
      }
      return 0;
    }
    // Expect delayed results extraction (used for sub-fragments) for estimator only;
    CHECK(results);
    std::vector<int64_t> reduced_outs;
    const auto num_frags = col_buffers.size();
    const size_t entry_count =
        device_type == ExecutorDeviceType::GPU
            ? (compilation_result.gpu_smem_context.isSharedMemoryUsed()
                   ? 1
                   : blockSize() * gridSize() * num_frags)
            : num_frags;
    if (size_t(1) == entry_count) {
      for (auto out : out_vec) {
        CHECK(out);
        reduced_outs.push_back(*out);
      }
    } else {
      size_t out_vec_idx = 0;

      for (const auto target_expr : ra_exe_unit.target_exprs) {
        const auto agg_info =
            get_target_info(target_expr, getConfig().exec.group_by.bigint_count);
        CHECK(agg_info.is_agg || target_expr->is<hdk::ir::Constant>())
            << target_expr->toString();

        int64_t val1;
        const bool float_argument_input = takes_float_argument(agg_info);
        if (is_distinct_target(agg_info) ||
            agg_info.agg_kind == hdk::ir::AggType::kApproxQuantile) {
          CHECK(agg_info.agg_kind == hdk::ir::AggType::kCount ||
                agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct ||
                agg_info.agg_kind == hdk::ir::AggType::kApproxQuantile);
          val1 = out_vec[out_vec_idx][0];
          error_code = 0;
        } else {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx));
          std::tie(val1, error_code) = Executor::reduceResults(
              agg_info.agg_kind,
              agg_info.type,
              query_exe_context->getAggInitValForIndex(out_vec_idx),
              float_argument_input ? sizeof(int32_t) : chosen_bytes,
              out_vec[out_vec_idx],
              entry_count,
              false,
              float_argument_input);
        }
        if (error_code) {
          break;
        }
        reduced_outs.push_back(val1);
        if (agg_info.agg_kind == hdk::ir::AggType::kAvg ||
            (agg_info.agg_kind == hdk::ir::AggType::kSample &&
             (agg_info.type->isString() || agg_info.type->isArray()))) {
          const auto chosen_bytes = static_cast<size_t>(
              query_exe_context->query_mem_desc_.getPaddedSlotWidthBytes(out_vec_idx +
                                                                         1));
          int64_t val2;
          std::tie(val2, error_code) = Executor::reduceResults(
              agg_info.agg_kind == hdk::ir::AggType::kAvg ? hdk::ir::AggType::kCount
                                                          : agg_info.agg_kind,
              agg_info.type,
              query_exe_context->getAggInitValForIndex(out_vec_idx + 1),
              float_argument_input ? sizeof(int32_t) : chosen_bytes,
              out_vec[out_vec_idx + 1],
              entry_count,
              false,
              false);
          if (error_code) {
            break;
          }
          reduced_outs.push_back(val2);
          ++out_vec_idx;
        }
        ++out_vec_idx;
      }
    }

    if (error_code) {
      return error_code;
    }

    CHECK_EQ(size_t(1), query_exe_context->query_buffers_->result_sets_.size());
    auto rows_ptr = std::shared_ptr<ResultSet>(
        query_exe_context->query_buffers_->result_sets_[0].release());
    rows_ptr->fillOneEntry(reduced_outs);
    *results = std::move(rows_ptr);
    return error_code;
  }

  RelAlgExecutionUnit ra_exe_unit_copy = ra_exe_unit;
  // For UNION ALL, filter out input_descs and input_col_descs that are not associated
  // with outer_table_id.
  if (ra_exe_unit_copy.union_all) {
    // Sort outer_table_id first, then pop the rest off of ra_exe_unit_copy.input_descs.
    std::stable_sort(ra_exe_unit_copy.input_descs.begin(),
                     ra_exe_unit_copy.input_descs.end(),
                     [outer_table_id](auto const& a, auto const& b) {
                       return a.getTableId() == outer_table_id &&
                              b.getTableId() != outer_table_id;
                     });
    while (!ra_exe_unit_copy.input_descs.empty() &&
           ra_exe_unit_copy.input_descs.back().getTableId() != outer_table_id) {
      ra_exe_unit_copy.input_descs.pop_back();
    }
    // Filter ra_exe_unit_copy.input_col_descs.
    ra_exe_unit_copy.input_col_descs.remove_if(
        [outer_table_id](auto const& input_col_desc) {
          return input_col_desc->getTableId() != outer_table_id;
        });
    query_exe_context->query_mem_desc_.setEntryCount(ra_exe_unit_copy.scan_limit);
  }

  if (device_type == ExecutorDeviceType::CPU) {
    const int32_t scan_limit_for_query =
        ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit;
    const int32_t max_matched = scan_limit_for_query == 0
                                    ? query_exe_context->query_mem_desc_.getEntryCount()
                                    : scan_limit_for_query;
    CpuCompilationContext* cpu_generated_code =
        dynamic_cast<CpuCompilationContext*>(compilation_result.generated_code.get());
    CHECK(cpu_generated_code);
    query_exe_context->launchCpuCode(ra_exe_unit_copy,
                                     cpu_generated_code,
                                     hoist_literals,
                                     hoist_buf,
                                     col_buffers,
                                     num_rows,
                                     frag_offsets,
                                     max_matched,
                                     &error_code,
                                     num_tables,
                                     join_hash_table_ptrs,
                                     rows_to_process);
  } else {
    try {
      CompilationContext* gpu_generated_code = compilation_result.generated_code.get();
      CHECK(gpu_generated_code);
      query_exe_context->launchGpuCode(
          ra_exe_unit_copy,
          gpu_generated_code,
          hoist_literals,
          hoist_buf,
          col_buffers,
          num_rows,
          frag_offsets,
          ra_exe_unit_copy.union_all ? ra_exe_unit_copy.scan_limit : scan_limit,
          data_mgr,
          getBufferProvider(),
          blockSize(),
          gridSize(),
          device_id,
          compilation_result.gpu_smem_context.getSharedMemorySize(),
          &error_code,
          num_tables,
          allow_runtime_interrupt,
          join_hash_table_ptrs);
    } catch (const OutOfMemory&) {
      return ERR_OUT_OF_GPU_MEM;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error launching the GPU kernel: " << e.what();
    }
  }

  if (error_code == Executor::ERR_OVERFLOW_OR_UNDERFLOW ||
      error_code == Executor::ERR_DIV_BY_ZERO ||
      error_code == Executor::ERR_OUT_OF_TIME ||
      error_code == Executor::ERR_INTERRUPTED ||
      error_code == Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES ||
      error_code == Executor::ERR_WIDTH_BUCKET_INVALID_ARGUMENT) {
    return error_code;
  }

  if (results && error_code != Executor::ERR_OVERFLOW_OR_UNDERFLOW &&
      error_code != Executor::ERR_DIV_BY_ZERO) {
    *results = query_exe_context->getRowSet(
        ra_exe_unit_copy, query_exe_context->query_mem_desc_, co);
    CHECK(*results);
    VLOG(2) << "results->rowCount()=" << (*results)->rowCount();
    (*results)->holdLiterals(hoist_buf);
  }
  if (results && error_code &&
      (!scan_limit || check_rows_less_than_needed(*results, scan_limit))) {
    return error_code;  // unlucky, not enough results and we ran out of slots
  }

  return 0;
}

std::vector<int64_t> Executor::getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                                    const int device_id) {
  std::vector<int64_t> table_ptrs;
  const auto& join_hash_tables = plan_state_->join_info_.join_hash_tables_;
  for (auto hash_table : join_hash_tables) {
    if (!hash_table) {
      CHECK(table_ptrs.empty());
      return {};
    }
    table_ptrs.push_back(hash_table->getJoinHashBuffer(
        device_type, device_type == ExecutorDeviceType::GPU ? device_id : 0));
  }
  return table_ptrs;
}

void Executor::nukeOldState(const bool allow_lazy_fetch,
                            const std::vector<InputTableInfo>& query_infos,
                            const RelAlgExecutionUnit* ra_exe_unit) {
  kernel_queue_time_ms_ = 0;
  compilation_queue_time_ms_ = 0;
  const bool contains_left_deep_outer_join =
      ra_exe_unit && std::find_if(ra_exe_unit->join_quals.begin(),
                                  ra_exe_unit->join_quals.end(),
                                  [](const JoinCondition& join_condition) {
                                    return join_condition.type == JoinType::LEFT;
                                  }) != ra_exe_unit->join_quals.end();
  cgen_state_.reset(new CgenState(query_infos.size(),
                                  contains_left_deep_outer_join,
                                  getConfig().debug.enable_automatic_ir_metadata,
                                  getExtensionModuleContext(),
                                  getContext()));
  plan_state_.reset(new PlanState(
      allow_lazy_fetch && !contains_left_deep_outer_join, query_infos, this));
}

void Executor::preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                                  const std::vector<InputTableInfo>& query_infos) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto ld_count = input_descs.size();
  auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
  for (size_t i = 0; i < ld_count; ++i) {
    CHECK_LT(i, query_infos.size());
    const auto frag_count = query_infos[i].info->fragments.size();
    if (i > 0) {
      cgen_state_->frag_offsets_.push_back(nullptr);
    } else {
      if (frag_count > 1) {
        cgen_state_->frag_offsets_.push_back(cgen_state_->ir_builder_.CreateLoad(
            get_int_type(64, cgen_state_->context_), frag_off_ptr));
      } else {
        cgen_state_->frag_offsets_.push_back(nullptr);
      }
    }
  }
}

Executor::JoinHashTableOrError Executor::buildHashTableForQualifier(
    const std::shared_ptr<const hdk::ir::BinOper>& qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map) {
  if (config_->exec.watchdog.enable_dynamic && interrupted_.load()) {
    throw QueryExecutionError(ERR_INTERRUPTED);
  }
  try {
    auto tbl = HashJoin::getInstance(qual_bin_oper,
                                     query_infos,
                                     memory_level,
                                     join_type,
                                     preferred_hash_type,
                                     deviceCountForMemoryLevel(memory_level),
                                     data_provider,
                                     column_cache,
                                     this,
                                     hashtable_build_dag_map,
                                     table_id_to_node_map);
    return {tbl, ""};
  } catch (const HashJoinFail& e) {
    return {nullptr, e.what()};
  }
}

int8_t Executor::warpSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return gpu_mgr->getSubGroupSize();
}

// TODO(adb): should these three functions have consistent symantics if cuda mgr does not
// exist?
unsigned Executor::gridSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return grid_size_x_ ? grid_size_x_ : gpu_mgr->getGridSize();
}

unsigned Executor::numBlocksPerMP() const {
  return grid_size_x_ ? std::ceil(grid_size_x_ / gpuMgr()->getMinEUNumForAllDevices())
                      : 2;
}

unsigned Executor::blockSize() const {
  CHECK(data_mgr_);
  const auto gpu_mgr = data_mgr_->getGpuMgr();
  if (!gpu_mgr) {
    return 0;
  }
  return block_size_x_ ? block_size_x_ : gpu_mgr->getMaxBlockSize();
}

size_t Executor::maxGpuSlabSize() const {
  return config_->mem.gpu.max_slab_size;
}

int64_t Executor::deviceCycles(int milliseconds) const {
  if (gpuMgr()->getPlatform() != GpuMgrPlatform::CUDA) {
    return 0;
  }
  const auto& dev_props = cudaMgr()->getAllDeviceProperties();
  return static_cast<int64_t>(dev_props.front().clockKhz) * milliseconds;
}

llvm::Value* Executor::castToFP(llvm::Value* value,
                                const hdk::ir::Type* from_type,
                                const hdk::ir::Type* to_type) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  if (value->getType()->isIntegerTy() && from_type->isNumber() &&
      to_type->isFloatingPoint() &&
      (!from_type->isFloatingPoint() || from_type->size() != to_type->size())) {
    llvm::Type* fp_type{nullptr};
    switch (to_type->size()) {
      case 4:
        fp_type = llvm::Type::getFloatTy(cgen_state_->context_);
        break;
      case 8:
        fp_type = llvm::Type::getDoubleTy(cgen_state_->context_);
        break;
      default:
        LOG(FATAL) << "Unsupported FP size: " << to_type->size();
    }
    value = cgen_state_->ir_builder_.CreateSIToFP(value, fp_type);
    if (from_type->isDecimal()) {
      auto scale = from_type->as<hdk::ir::DecimalType>()->scale();
      if (scale) {
        value = cgen_state_->ir_builder_.CreateFDiv(
            value, llvm::ConstantFP::get(value->getType(), exp_to_scale(scale)));
      }
    }
  }
  return value;
}

#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "DateAdd.cpp"
#include "StringFunctions.cpp"
#undef EXECUTE_INCLUDE

namespace {
// Note(Wamsi): `get_hpt_overflow_underflow_safe_scaled_value` will return `true` for safe
// scaled epoch value and `false` for overflow/underflow values as the first argument of
// return type.
std::tuple<bool, int64_t, int64_t> get_hpt_overflow_underflow_safe_scaled_values(
    const int64_t chunk_min,
    const int64_t chunk_max,
    const hdk::ir::Type* lhs_type,
    const hdk::ir::Type* rhs_type) {
  auto lunit = lhs_type->isTimestamp() ? lhs_type->as<hdk::ir::TimestampType>()->unit()
                                       : hdk::ir::TimeUnit::kSecond;
  auto runit = rhs_type->isTimestamp() ? rhs_type->as<hdk::ir::TimestampType>()->unit()
                                       : hdk::ir::TimeUnit::kSecond;
  CHECK(lunit != runit);
  if (lunit > runit) {
    auto scale = hdk::ir::unitsPerSecond(lunit) / hdk::ir::unitsPerSecond(runit);
    // LHS type precision is more than RHS col type. No chance of overflow/underflow.
    return {true, chunk_min / scale, chunk_max / scale};
  }

  auto scale = hdk::ir::unitsPerSecond(runit) / hdk::ir::unitsPerSecond(lunit);
  using checked_int64_t = boost::multiprecision::number<
      boost::multiprecision::cpp_int_backend<64,
                                             64,
                                             boost::multiprecision::signed_magnitude,
                                             boost::multiprecision::checked,
                                             void>>;

  try {
    auto ret =
        std::make_tuple(true,
                        int64_t(checked_int64_t(chunk_min) * checked_int64_t(scale)),
                        int64_t(checked_int64_t(chunk_max) * checked_int64_t(scale)));
    return ret;
  } catch (const std::overflow_error& e) {
    // noop
  }
  return std::make_tuple(false, chunk_min, chunk_max);
}

}  // namespace

FragmentSkipStatus Executor::canSkipFragmentForFpQual(
    const hdk::ir::BinOper* comp_expr,
    const hdk::ir::ColumnVar* lhs_col,
    const FragmentInfo& fragment,
    const hdk::ir::Constant* rhs_const) const {
  const int col_id = lhs_col->columnId();
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
    return FragmentSkipStatus::NOT_SKIPPABLE;
  }
  double chunk_min{0.};
  double chunk_max{0.};
  const auto& chunk_type = lhs_col->type();
  chunk_min = extract_min_stat_fp_type(chunk_meta_it->second->chunkStats(), chunk_type);
  chunk_max = extract_max_stat_fp_type(chunk_meta_it->second->chunkStats(), chunk_type);
  if (chunk_min > chunk_max) {
    return FragmentSkipStatus::INVALID;
  }

  const auto datum_fp = rhs_const->value();
  const auto rhs_type = rhs_const->type();
  CHECK(rhs_type->isFloatingPoint());

  // Do we need to codegen the constant like the integer path does?
  const auto rhs_val = rhs_type->isFp32() ? datum_fp.floatval : datum_fp.doubleval;

  // Todo: dedup the following comparison code with the integer/timestamp path, it is
  // slightly tricky due to do cleanly as we do not have rowid on this path
  switch (comp_expr->opType()) {
    case hdk::ir::OpType::kGe:
      if (chunk_max < rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case hdk::ir::OpType::kGt:
      if (chunk_max <= rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case hdk::ir::OpType::kLe:
      if (chunk_min > rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case hdk::ir::OpType::kLt:
      if (chunk_min >= rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    case hdk::ir::OpType::kEq:
      if (chunk_min > rhs_val || chunk_max < rhs_val) {
        return FragmentSkipStatus::SKIPPABLE;
      }
      break;
    default:
      break;
  }
  return FragmentSkipStatus::NOT_SKIPPABLE;
}

std::pair<bool, int64_t> Executor::skipFragment(
    const InputDescriptor& table_desc,
    const FragmentInfo& fragment,
    const std::list<hdk::ir::ExprPtr>& simple_quals,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx,
    compiler::CodegenTraitsDescriptor cgen_traits_desc) {
  const int db_id = table_desc.getDatabaseId();
  const int table_id = table_desc.getTableId();

  for (const auto& simple_qual : simple_quals) {
    const auto comp_expr = std::dynamic_pointer_cast<const hdk::ir::BinOper>(simple_qual);
    if (!comp_expr) {
      // is this possible?
      return {false, -1};
    }
    const auto lhs = comp_expr->leftOperand();
    auto lhs_col = dynamic_cast<const hdk::ir::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->tableId() || lhs_col->rteIdx()) {
      // See if lhs is a simple cast that was allowed through normalize_simple_predicate
      auto lhs_uexpr = dynamic_cast<const hdk::ir::UOper*>(lhs);
      if (lhs_uexpr) {
        CHECK(lhs_uexpr->isCast());  // We should have only been passed a cast expression
        lhs_col = dynamic_cast<const hdk::ir::ColumnVar*>(lhs_uexpr->operand());
        if (!lhs_col || !lhs_col->tableId() || lhs_col->rteIdx()) {
          continue;
        }
      } else {
        continue;
      }
    }
    const auto rhs = comp_expr->rightOperand();
    const auto rhs_const = dynamic_cast<const hdk::ir::Constant*>(rhs);
    if (!rhs_const) {
      // is this possible?
      return {false, -1};
    }
    if (!lhs->type()->isInteger() && !lhs->type()->isDateTime() &&
        !lhs->type()->isFloatingPoint()) {
      continue;
    }

    if (lhs->type()->isFloatingPoint()) {
      const auto fragment_skip_status =
          canSkipFragmentForFpQual(comp_expr.get(), lhs_col, fragment, rhs_const);
      switch (fragment_skip_status) {
        case FragmentSkipStatus::SKIPPABLE:
          return {true, -1};
        case FragmentSkipStatus::INVALID:
          return {false, -1};
        case FragmentSkipStatus::NOT_SKIPPABLE:
          continue;
        default:
          UNREACHABLE();
      }
    }

    // Everything below is logic for integer and integer-backed timestamps
    // TODO: Factor out into separate function per canSkipFragmentForFpQual above

    const int col_id = lhs_col->columnId();
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    int64_t chunk_min{0};
    int64_t chunk_max{0};
    bool is_rowid{false};
    size_t start_rowid{0};
    if (chunk_meta_it == fragment.getChunkMetadataMap().end()) {
      if (lhs_col->isVirtual()) {
        const auto& table_generation = getTableGeneration(db_id, table_id);
        start_rowid = table_generation.start_rowid;
        chunk_min = frag_offsets[frag_idx] + start_rowid;
        chunk_max = frag_offsets[frag_idx + 1] - 1 + start_rowid;
        is_rowid = true;
      }
    } else {
      const auto& chunk_type = lhs_col->type();
      chunk_min =
          extract_min_stat_int_type(chunk_meta_it->second->chunkStats(), chunk_type);
      chunk_max =
          extract_max_stat_int_type(chunk_meta_it->second->chunkStats(), chunk_type);
    }
    if (chunk_min > chunk_max) {
      // invalid metadata range, do not skip fragment
      return {false, -1};
    }
    auto lhs_col_unit = lhs_col->type()->isTimestamp()
                            ? lhs_col->type()->as<hdk::ir::TimestampType>()->unit()
                            : hdk::ir::TimeUnit::kSecond;
    auto rhs_unit = rhs->type()->isTimestamp()
                        ? rhs->type()->as<hdk::ir::TimestampType>()->unit()
                        : hdk::ir::TimeUnit::kSecond;
    if (lhs->type()->isTimestamp() && (lhs_col_unit != rhs_unit) &&
        (lhs_col_unit > hdk::ir::TimeUnit::kSecond ||
         rhs_unit > hdk::ir::TimeUnit::kSecond)) {
      // If original timestamp lhs col has different precision,
      // column metadata holds value in original precision
      // therefore adjust rhs value to match lhs precision

      // Note(Wamsi): We adjust rhs const value instead of lhs value to not
      // artificially limit the lhs column range. RHS overflow/underflow is already
      // been validated in `TimeGM::get_overflow_underflow_safe_epoch`.
      bool is_valid;
      std::tie(is_valid, chunk_min, chunk_max) =
          get_hpt_overflow_underflow_safe_scaled_values(
              chunk_min, chunk_max, lhs_col->type(), rhs_const->type());
      if (!is_valid) {
        VLOG(4) << "Overflow/Underflow detecting in fragments skipping logic.\nChunk min "
                   "value: "
                << std::to_string(chunk_min)
                << "\nChunk max value: " << std::to_string(chunk_max)
                << "\nLHS col precision is: " << toString(lhs_col_unit)
                << "\nRHS precision is: " << toString(rhs_unit) << ".";
        return {false, -1};
      }
    }
    if (lhs_col->type()->isTimestamp() && rhs_const->type()->isDate()) {
      // It is obvious that a cast from timestamp to date is happening here,
      // so we have to correct the chunk min and max values to lower the precision as of
      // the date
      chunk_min = truncate_high_precision_timestamp_to_date(
          chunk_min, hdk::ir::unitsPerSecond(lhs_col_unit));
      chunk_max = truncate_high_precision_timestamp_to_date(
          chunk_max, hdk::ir::unitsPerSecond(lhs_col_unit));
    }
    llvm::LLVMContext local_context;
    CgenState local_cgen_state(getConfig(), local_context);
    CodeGenerator code_generator(
        getConfig(), &local_cgen_state, nullptr, cgen_traits_desc);

    const auto rhs_val =
        CodeGenerator::codegenIntConst(rhs_const, &local_cgen_state)->getSExtValue();

    switch (comp_expr->opType()) {
      case hdk::ir::OpType::kGe:
        if (chunk_max < rhs_val) {
          return {true, -1};
        }
        break;
      case hdk::ir::OpType::kGt:
        if (chunk_max <= rhs_val) {
          return {true, -1};
        }
        break;
      case hdk::ir::OpType::kLe:
        if (chunk_min > rhs_val) {
          return {true, -1};
        }
        break;
      case hdk::ir::OpType::kLt:
        if (chunk_min >= rhs_val) {
          return {true, -1};
        }
        break;
      case hdk::ir::OpType::kEq:
        if (chunk_min > rhs_val || chunk_max < rhs_val) {
          return {true, -1};
        } else if (is_rowid) {
          return {false, rhs_val - start_rowid};
        }
        break;
      default:
        break;
    }
  }
  return {false, -1};
}

/*
 *   The skipFragmentInnerJoins process all quals stored in the execution unit's
 * join_quals and gather all the ones that meet the "simple_qual" characteristics
 * (logical expressions with AND operations, etc.). It then uses the skipFragment function
 * to decide whether the fragment should be skipped or not. The fragment will be skipped
 * if at least one of these skipFragment calls return a true statment in its first value.
 *   - The code depends on skipFragment's output to have a meaningful (anything but -1)
 * second value only if its first value is "false".
 *   - It is assumed that {false, n  > -1} has higher priority than {true, -1},
 *     i.e., we only skip if none of the quals trigger the code to update the
 * rowid_lookup_key
 *   - Only AND operations are valid and considered:
 *     - `select * from t1,t2 where A and B and C`: A, B, and C are considered for causing
 * the skip
 *     - `select * from t1,t2 where (A or B) and C`: only C is considered
 *     - `select * from t1,t2 where A or B`: none are considered (no skipping).
 *   - NOTE: (re: intermediate projections) the following two queries are fundamentally
 * implemented differently, which cause the first one to skip correctly, but the second
 * one will not skip.
 *     -  e.g. #1, select * from t1 join t2 on (t1.i=t2.i) where (A and B); -- skips if
 * possible
 *     -  e.g. #2, select * from t1 join t2 on (t1.i=t2.i and A and B); -- intermediate
 * projection, no skipping
 */
std::pair<bool, int64_t> Executor::skipFragmentInnerJoins(
    const InputDescriptor& table_desc,
    const RelAlgExecutionUnit& ra_exe_unit,
    const FragmentInfo& fragment,
    const std::vector<uint64_t>& frag_offsets,
    const size_t frag_idx,
    compiler::CodegenTraitsDescriptor cgen_traits_desc) {
  std::pair<bool, int64_t> skip_frag{false, -1};
  for (auto& inner_join : ra_exe_unit.join_quals) {
    if (inner_join.type != JoinType::INNER) {
      continue;
    }

    // extracting all the conjunctive simple_quals from the quals stored for the inner
    // join
    std::list<hdk::ir::ExprPtr> inner_join_simple_quals;
    for (auto& qual : inner_join.quals) {
      auto temp_qual = qual_to_conjunctive_form(qual);
      inner_join_simple_quals.insert(inner_join_simple_quals.begin(),
                                     temp_qual.simple_quals.begin(),
                                     temp_qual.simple_quals.end());
    }
    auto temp_skip_frag = skipFragment(table_desc,
                                       fragment,
                                       inner_join_simple_quals,
                                       frag_offsets,
                                       frag_idx,
                                       cgen_traits_desc);
    if (temp_skip_frag.second != -1) {
      skip_frag.second = temp_skip_frag.second;
      return skip_frag;
    } else {
      skip_frag.first = skip_frag.first || temp_skip_frag.first;
    }
  }
  return skip_frag;
}

AggregatedColRange Executor::computeColRangesCache(
    const std::unordered_set<InputColDescriptor>& col_descs) {
  AggregatedColRange agg_col_range_cache;
  TableRefSet phys_table_refs;
  for (const auto& col_desc : col_descs) {
    phys_table_refs.insert({col_desc.getDatabaseId(), col_desc.getTableId()});
  }
  std::vector<InputTableInfo> query_infos;
  for (auto& tref : phys_table_refs) {
    query_infos.emplace_back(InputTableInfo{
        tref.db_id, tref.table_id, getTableInfo(tref.db_id, tref.table_id)});
  }
  for (const auto& col_desc : col_descs) {
    if (ExpressionRange::typeSupportsRange(col_desc.type())) {
      const auto col_var = std::make_unique<hdk::ir::ColumnVar>(col_desc.getColInfo(), 0);
      const auto col_range = getLeafColumnRange(col_var.get(), query_infos, this, false);
      agg_col_range_cache.setColRange(
          {col_desc.getColId(), col_desc.getTableId(), col_desc.getDatabaseId()},
          col_range);
    }
  }
  return agg_col_range_cache;
}

StringDictionaryGenerations Executor::computeStringDictionaryGenerations(
    const std::unordered_set<InputColDescriptor>& col_descs) {
  StringDictionaryGenerations string_dictionary_generations;
  for (const auto& col_desc : col_descs) {
    auto col_type = col_desc.type()->isArray()
                        ? col_desc.type()->as<hdk::ir::ArrayBaseType>()->elemType()
                        : col_desc.type();
    if (col_type->isExtDictionary()) {
      const int dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();
      const auto dd = data_mgr_->getDictMetadata(dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(dict_id, dd->stringDict->entryCount());
    }
  }
  return string_dictionary_generations;
}

TableGenerations Executor::computeTableGenerations(
    std::unordered_set<std::pair<int, int>> phys_table_ids) {
  TableGenerations table_generations;
  for (auto [db_id, table_id] : phys_table_ids) {
    table_generations.setGeneration(
        db_id,
        table_id,
        TableGeneration{
            static_cast<int64_t>(getTableInfo(db_id, table_id)->getPhysicalNumTuples()),
            0});
  }
  return table_generations;
}

void Executor::setupCaching(
    DataProvider* data_provider,
    const std::unordered_set<InputColDescriptor>& col_descs,
    const std::unordered_set<std::pair<int, int>>& phys_table_ids) {
  row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>(
      data_provider, Executor::getArenaBlockSize(), cpu_threads());
  string_dictionary_generations_ = computeStringDictionaryGenerations(col_descs);
  agg_col_range_cache_ = computeColRangesCache(col_descs);
  table_generations_ = computeTableGenerations(phys_table_ids);
}

mapd_shared_mutex& Executor::getDataRecyclerLock() {
  return recycler_mutex_;
}

QueryPlanDagCache& Executor::getQueryPlanDagCache() {
  return *query_plan_dag_cache_;
}

JoinColumnsInfo Executor::getJoinColumnsInfo(const hdk::ir::Expr* join_expr,
                                             JoinColumnSide target_side,
                                             bool extract_only_col_id) {
  return query_plan_dag_cache_->getJoinColumnsInfoString(
      join_expr, target_side, extract_only_col_id);
}

void Executor::addToCardinalityCache(const std::string& cache_key,
                                     const size_t cache_value) {
  if (config_->cache.use_estimator_result_cache) {
    mapd_unique_lock<mapd_shared_mutex> lock(recycler_mutex_);
    cardinality_cache_[cache_key] = cache_value;
    VLOG(1) << "Put estimated cardinality to the cache";
  }
}

Executor::CachedCardinality Executor::getCachedCardinality(const std::string& cache_key) {
  mapd_shared_lock<mapd_shared_mutex> lock(recycler_mutex_);
  if (config_->cache.use_estimator_result_cache &&
      cardinality_cache_.find(cache_key) != cardinality_cache_.end()) {
    VLOG(1) << "Reuse cached cardinality";
    return {true, cardinality_cache_[cache_key]};
  }
  return {false, -1};
}

bool Executor::checkNonKernelTimeInterrupted() const {
  // this function should be called within an executor which is assigned
  // to the specific query thread (that indicates we already enroll the session)
  // check whether this is called from non unitary executor
  return interrupted_.load();
}

const std::unique_ptr<llvm::Module>& ExtensionModuleContext::getRTUdfModule(
    bool is_gpu) const {
  std::shared_lock lock(Executor::register_runtime_extension_functions_mutex_);
  return getExtensionModule(
      (is_gpu ? ExtModuleKinds::rt_udf_gpu_module : ExtModuleKinds::rt_udf_cpu_module));
}

mapd_shared_mutex Executor::execute_mutex_;

std::mutex Executor::gpu_active_modules_mutex_;
uint32_t Executor::gpu_active_modules_device_mask_{0x0};
void* Executor::gpu_active_modules_[max_gpu_count];

std::shared_mutex Executor::register_runtime_extension_functions_mutex_;
std::mutex Executor::kernel_mutex_;
std::atomic<size_t> Executor::executor_id_ctr_{0};

std::unique_ptr<QueryPlanDagCache> Executor::query_plan_dag_cache_;
std::once_flag Executor::first_init_flag_;
mapd_shared_mutex Executor::recycler_mutex_;
std::unordered_map<std::string, size_t> Executor::cardinality_cache_;
