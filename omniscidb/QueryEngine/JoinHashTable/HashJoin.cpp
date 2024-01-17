/*
 * Copyright 2019 MapD Technologies, Inc.
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

#include "QueryEngine/JoinHashTable/HashJoin.h"

#include "IR/ExprCollector.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RuntimeFunctions.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

//! fetchJoinColumn() calls ColumnFetcher::makeJoinColumn(), then copies the
//! JoinColumn's col_chunks_buff memory onto the GPU if required by the
//! effective_memory_level parameter. The dev_buff_owner parameter will
//! manage the GPU memory.
JoinColumn HashJoin::fetchJoinColumn(
    const hdk::ir::ColumnVar* hash_col,
    const std::vector<FragmentInfo>& fragment_info,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    DeviceAllocator* dev_buff_owner,
    std::vector<std::shared_ptr<void>>& malloc_owner,
    Executor* executor,
    ColumnCacheMap* column_cache) {
  static std::mutex fragment_fetch_mutex;
  std::lock_guard<std::mutex> fragment_fetch_lock(fragment_fetch_mutex);
  try {
    JoinColumn join_column = ColumnFetcher::makeJoinColumn(executor,
                                                           *hash_col,
                                                           fragment_info,
                                                           effective_memory_level,
                                                           device_id,
                                                           dev_buff_owner,
                                                           chunks_owner,
                                                           malloc_owner,
                                                           data_provider_,
                                                           *column_cache);
    if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
      CHECK(dev_buff_owner);
      auto device_col_chunks_buff = dev_buff_owner->alloc(join_column.col_chunks_buff_sz);
      dev_buff_owner->copyToDevice(device_col_chunks_buff,
                                   join_column.col_chunks_buff,
                                   join_column.col_chunks_buff_sz);
      join_column.col_chunks_buff = device_col_chunks_buff;
    }
    return join_column;
  } catch (...) {
    throw FailedToFetchColumn();
  }
}

namespace {

template <typename T>
std::string toStringFlat(const HashJoin* hash_table,
                         const ExecutorDeviceType device_type,
                         const int device_id) {
  auto mem =
      reinterpret_cast<const T*>(hash_table->getJoinHashBuffer(device_type, device_id));
  auto memsz = hash_table->getJoinHashBufferSize(device_type, device_id) / sizeof(T);
  std::string txt;
  for (size_t i = 0; i < memsz; ++i) {
    if (i > 0) {
      txt += ", ";
    }
    txt += std::to_string(mem[i]);
  }
  return txt;
}

}  // anonymous namespace

std::string HashJoin::toStringFlat64(const ExecutorDeviceType device_type,
                                     const int device_id) const {
  return toStringFlat<int64_t>(this, device_type, device_id);
}

std::string HashJoin::toStringFlat32(const ExecutorDeviceType device_type,
                                     const int device_id) const {
  return toStringFlat<int32_t>(this, device_type, device_id);
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferEntry& e) {
  os << "  {{";
  bool first = true;
  for (auto k : e.key) {
    if (!first) {
      os << ",";
    } else {
      first = false;
    }
    os << k;
  }
  os << "}, ";
  os << "{";
  first = true;
  for (auto p : e.payload) {
    if (!first) {
      os << ", ";
    } else {
      first = false;
    }
    os << p;
  }
  os << "}}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferSet& s) {
  os << "{\n";
  bool first = true;
  for (auto e : s) {
    if (!first) {
      os << ",\n";
    } else {
      first = false;
    }
    os << e;
  }
  if (!s.empty()) {
    os << "\n";
  }
  os << "}\n";
  return os;
}

HashJoinMatchingSet HashJoin::codegenMatchingSet(
    const std::vector<llvm::Value*>& hash_join_idx_args_in,
    const bool col_is_nullable,
    const bool is_bw_eq,
    const int64_t sub_buff_size,
    Executor* executor,
    const CompilationOptions& co,
    bool is_bucketized) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  using namespace std::string_literals;

  std::string fname(is_bucketized ? "bucketized_hash_join_idx"s : "hash_join_idx"s);
  compiler::CodegenTraits cgen_traits =
      compiler::CodegenTraits::get(co.codegen_traits_desc);

  if (is_bw_eq) {
    fname += "_bitwise";
  }
  if (!is_bw_eq && col_is_nullable) {
    fname += "_nullable";
  }

  const auto slot_lv = executor->cgen_state_->emitCall(fname, hash_join_idx_args_in);
  const auto slot_valid_lv = executor->cgen_state_->ir_builder_.CreateICmpSGE(
      slot_lv, executor->cgen_state_->llInt(int64_t(0)));

  auto pos_ptr = hash_join_idx_args_in[0];
  CHECK(pos_ptr);

  auto count_ptr = executor->cgen_state_->ir_builder_.CreateAdd(
      pos_ptr, executor->cgen_state_->llInt(sub_buff_size));
  auto hash_join_idx_args = hash_join_idx_args_in;
  hash_join_idx_args[0] = executor->cgen_state_->ir_builder_.CreatePtrToInt(
      count_ptr, llvm::Type::getInt64Ty(executor->cgen_state_->context_));

  const auto row_count_lv = executor->cgen_state_->ir_builder_.CreateSelect(
      slot_valid_lv,
      executor->cgen_state_->emitCall(fname, hash_join_idx_args),
      executor->cgen_state_->llInt(int64_t(0)));
  auto rowid_base_i32 = executor->cgen_state_->ir_builder_.CreateIntToPtr(
      executor->cgen_state_->ir_builder_.CreateAdd(
          pos_ptr, executor->cgen_state_->llInt(2 * sub_buff_size)),
      llvm::Type::getInt32PtrTy(executor->cgen_state_->context_,
                                cgen_traits.getLocalAddrSpace()));

  auto rowid_ptr_i32 = executor->cgen_state_->ir_builder_.CreateGEP(
      get_int_type(32, executor->cgen_state_->context_), rowid_base_i32, slot_lv);
  return {rowid_ptr_i32, row_count_lv, slot_lv};
}

llvm::Value* HashJoin::codegenHashTableLoad(const size_t table_idx, Executor* executor) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  llvm::Value* hash_ptr = nullptr;
  const auto total_table_count =
      executor->plan_state_->join_info_.join_hash_tables_.size();
  CHECK_LT(table_idx, total_table_count);

  // TODO: ambiguous hash_ptr, needs further investigation
  if (total_table_count > 1) {
    auto hash_tables_ptr =
        get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
    auto hash_pptr =
        table_idx > 0 ? executor->cgen_state_->ir_builder_.CreateGEP(
                            llvm::PointerType::get(
                                executor->cgen_state_->context_,
                                hash_tables_ptr->getType()->getPointerAddressSpace()),
                            hash_tables_ptr,
                            executor->cgen_state_->llInt(static_cast<int64_t>(table_idx)))
                      : hash_tables_ptr;
    hash_ptr = executor->cgen_state_->ir_builder_.CreateLoad(
        get_int_type(64, executor->cgen_state_->context_), hash_pptr, "hash_tables_ptr");
  } else {
    hash_ptr = get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
  }
  CHECK(hash_ptr);
  return hash_ptr;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<HashJoin> HashJoin::getInstance(
    const std::shared_ptr<const hdk::ir::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    const int device_count,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map) {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<HashJoin> join_hash_table;
  CHECK_GT(device_count, 0);
  if (dynamic_cast<const hdk::ir::ExpressionTuple*>(qual_bin_oper->leftOperand())) {
    VLOG(1) << "Trying to build keyed hash table:";
    join_hash_table = BaselineJoinHashTable::getInstance(qual_bin_oper,
                                                         query_infos,
                                                         memory_level,
                                                         join_type,
                                                         preferred_hash_type,
                                                         device_count,
                                                         data_provider,
                                                         column_cache,
                                                         executor,
                                                         hashtable_build_dag_map,
                                                         table_id_to_node_map);
  } else {
    try {
      VLOG(1) << "Trying to build perfect hash table:";
      join_hash_table = PerfectJoinHashTable::getInstance(qual_bin_oper,
                                                          query_infos,
                                                          memory_level,
                                                          join_type,
                                                          preferred_hash_type,
                                                          device_count,
                                                          data_provider,
                                                          column_cache,
                                                          executor,
                                                          hashtable_build_dag_map,
                                                          table_id_to_node_map);
    } catch (TooManyHashEntries&) {
      const auto join_quals = coalesce_singleton_equi_join(qual_bin_oper);
      CHECK_EQ(join_quals.size(), size_t(1));
      const auto join_qual =
          std::dynamic_pointer_cast<const hdk::ir::BinOper>(join_quals.front());
      VLOG(1) << "Trying to build keyed hash table after perfect hash table:";
      join_hash_table = BaselineJoinHashTable::getInstance(join_qual,
                                                           query_infos,
                                                           memory_level,
                                                           join_type,
                                                           preferred_hash_type,
                                                           device_count,
                                                           data_provider,
                                                           column_cache,
                                                           executor,
                                                           hashtable_build_dag_map,
                                                           table_id_to_node_map);
    }
  }
  CHECK(join_hash_table);
  if (VLOGGING(2)) {
    if (join_hash_table->getMemoryLevel() == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      for (int device_id = 0; device_id < join_hash_table->getDeviceCount();
           ++device_id) {
        if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::GPU, device_id) <=
            1000) {
          VLOG(2) << "Built GPU hash table: "
                  << join_hash_table->toString(ExecutorDeviceType::GPU, device_id);
        }
      }
    } else {
      if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::CPU) <= 1000) {
        VLOG(2) << "Built CPU hash table: "
                << join_hash_table->toString(ExecutorDeviceType::CPU);
      }
    }
  }
  return join_hash_table;
}

std::pair<const StringDictionary*, const StringDictionary*> HashJoin::getStrDictProxies(
    const InnerOuter& cols,
    const Executor* executor) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  auto inner_type = inner_col->type();
  const auto outer_col = dynamic_cast<const hdk::ir::ColumnVar*>(cols.second);
  std::pair<const StringDictionary*, const StringDictionary*>
      inner_outer_str_dict_proxies{nullptr, nullptr};
  if (inner_type->isExtDictionary() && outer_col) {
    CHECK(outer_col->type()->isExtDictionary());
    auto inner_dict_id = inner_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto outer_dict_id = outer_col->type()->as<hdk::ir::ExtDictionaryType>()->dictId();
    inner_outer_str_dict_proxies.first =
        executor->getStringDictionary(inner_dict_id, true);
    CHECK(inner_outer_str_dict_proxies.first);
    inner_outer_str_dict_proxies.second =
        executor->getStringDictionary(outer_dict_id, true);
    CHECK(inner_outer_str_dict_proxies.second);
    if (*inner_outer_str_dict_proxies.first == *inner_outer_str_dict_proxies.second) {
      // Dictionaries are the same - don't need to translate
      CHECK(inner_dict_id == outer_dict_id);
      inner_outer_str_dict_proxies.first = nullptr;
      inner_outer_str_dict_proxies.second = nullptr;
    }
  }
  return inner_outer_str_dict_proxies;
}

const std::vector<int32_t>* HashJoin::translateInnerToOuterStrDictProxies(
    const InnerOuter& cols,
    const Executor* executor) {
  const auto inner_outer_proxies = HashJoin::getStrDictProxies(cols, executor);
  const bool translate_dictionary =
      inner_outer_proxies.first && inner_outer_proxies.second;
  if (translate_dictionary) {
    const auto inner_dict_id =
        inner_outer_proxies.first->getBaseDictionary()->getDictId();
    const auto outer_dict_id =
        inner_outer_proxies.second->getBaseDictionary()->getDictId();
    CHECK_NE(inner_dict_id, outer_dict_id);
    return executor->getIntersectionStringProxyTranslationMap(
        inner_outer_proxies.first,
        inner_outer_proxies.second,
        executor->getRowSetMemoryOwner());
  }
  return nullptr;
}

int64_t HashJoin::getJoinHashBuffer(const ExecutorDeviceType device_type,
                                    const int device_id) const {
  // TODO: just make device_id a size_t
  CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
  if (!hash_tables_for_device_[device_id]) {
    return 0;
  }
  CHECK(hash_tables_for_device_[device_id]);
  auto hash_table = hash_tables_for_device_[device_id].get();
#ifdef HAVE_CUDA
  if (device_type == ExecutorDeviceType::CPU) {
    return reinterpret_cast<int64_t>(hash_table->getCpuBuffer());
  } else {
    CHECK(hash_table);
    const auto gpu_buff = hash_table->getGpuBuffer();
    return reinterpret_cast<CUdeviceptr>(gpu_buff);
  }
#else
  CHECK(device_type == ExecutorDeviceType::CPU);
  return reinterpret_cast<int64_t>(hash_table->getCpuBuffer());
#endif
}

CompositeKeyInfo HashJoin::getCompositeKeyInfo(
    const std::vector<InnerOuter>& inner_outer_pairs,
    const Executor* executor) {
  CHECK(executor);
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<const void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto outer_col = inner_outer_pair.second;
    auto inner_type = inner_col->type();
    auto outer_type = outer_col->type();
    ChunkKey cache_key_chunks_for_column{
        inner_col->dbId(), inner_col->tableId(), inner_col->columnId()};
    auto inner_dict_id = inner_type->isExtDictionary()
                             ? inner_type->as<hdk::ir::ExtDictionaryType>()->dictId()
                             : -1;
    auto outer_dict_id = outer_type->isExtDictionary()
                             ? outer_type->as<hdk::ir::ExtDictionaryType>()->dictId()
                             : -1;
    if (inner_type->isExtDictionary() && inner_dict_id != outer_dict_id) {
      CHECK(outer_type->isExtDictionary());
      const auto sd_inner_proxy = executor->getStringDictionaryProxy(
          inner_dict_id, executor->getRowSetMemoryOwner(), true);
      const auto sd_outer_proxy = executor->getStringDictionaryProxy(
          outer_dict_id, executor->getRowSetMemoryOwner(), true);
      CHECK(sd_inner_proxy && sd_outer_proxy);
      sd_inner_proxy_per_key.push_back(sd_inner_proxy);
      sd_outer_proxy_per_key.push_back(sd_outer_proxy);
      cache_key_chunks_for_column.push_back(sd_outer_proxy->getBaseGeneration());
    } else {
      sd_inner_proxy_per_key.emplace_back();
      sd_outer_proxy_per_key.emplace_back();
    }
    cache_key_chunks.push_back(cache_key_chunks_for_column);
  }
  return {sd_inner_proxy_per_key, sd_outer_proxy_per_key, cache_key_chunks};
}

std::vector<const std::vector<int32_t>*> HashJoin::translateCompositeStrDictProxies(
    const CompositeKeyInfo& composite_key_info,
    const Executor* executor) {
  const auto& inner_proxies = composite_key_info.sd_inner_proxy_per_key;
  const auto& outer_proxies = composite_key_info.sd_outer_proxy_per_key;
  const size_t num_proxies = inner_proxies.size();
  CHECK_EQ(num_proxies, outer_proxies.size());
  std::vector<const std::vector<int32_t>*> proxy_translation_maps;
  proxy_translation_maps.reserve(num_proxies);
  for (size_t proxy_pair_idx = 0; proxy_pair_idx < num_proxies; ++proxy_pair_idx) {
    const bool translate_proxies =
        inner_proxies[proxy_pair_idx] && outer_proxies[proxy_pair_idx];
    if (translate_proxies) {
      const auto inner_proxy =
          reinterpret_cast<const StringDictionary*>(inner_proxies[proxy_pair_idx]);
      const auto outer_proxy =
          reinterpret_cast<const StringDictionary*>(outer_proxies[proxy_pair_idx]);
      CHECK(inner_proxy);
      CHECK(outer_proxy);

      CHECK_NE(inner_proxy->getBaseDictionary()->getDictId(),
               outer_proxy->getBaseDictionary()->getDictId());
      proxy_translation_maps.emplace_back(
          executor->getIntersectionStringProxyTranslationMap(
              inner_proxy, outer_proxy, executor->getRowSetMemoryOwner()));
    } else {
      proxy_translation_maps.emplace_back(nullptr);
    }
  }
  return proxy_translation_maps;
}

std::shared_ptr<const hdk::ir::ColumnVar> getSyntheticColumnVar(int db_id,
                                                                std::string_view table,
                                                                std::string_view column,
                                                                int rte_idx,
                                                                Executor* executor) {
  auto schema_provider = executor->getSchemaProvider();
  auto table_info = schema_provider->getTableInfo(db_id, std::string(table));
  CHECK(table_info);
  auto col_info = schema_provider->getColumnInfo(*table_info, std::string(column));
  CHECK(col_info);

  auto cv = std::make_shared<hdk::ir::ColumnVar>(col_info, rte_idx);
  return cv;
}

class AllColumnVarsCollector
    : public hdk::ir::ExprCollector<std::set<const hdk::ir::ColumnVar*>,
                                    AllColumnVarsCollector> {
 protected:
  void visitColumnVar(const hdk::ir::ColumnVar* column) override {
    result_.insert(column);
  }
};

void setupSyntheticCaching(DataProvider* data_provider,
                           std::set<const hdk::ir::ColumnVar*> cvs,
                           Executor* executor) {
  std::unordered_set<std::pair<int, int>> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert({cv->dbId(), cv->tableId()});
  }

  std::unordered_set<InputColDescriptor> col_descs;
  for (auto cv : cvs) {
    col_descs.emplace(InputColDescriptor{cv->columnInfo(), cv->rteIdx()});
  }

  executor->setupCaching(data_provider, col_descs, phys_table_ids);
}

std::vector<InputTableInfo> getSyntheticInputTableInfo(
    std::set<const hdk::ir::ColumnVar*> cvs,
    Executor* executor) {
  std::unordered_set<std::pair<int, int>> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert({cv->dbId(), cv->tableId()});
  }

  // NOTE(sy): This vector ordering seems to work for now, but maybe we need to
  // review how rte_idx is assigned for ColumnVars. See for example Analyzer.h
  // and RelAlgExecutor.cpp and rte_idx there.
  std::vector<InputTableInfo> query_infos;
  for (auto [db_id, table_id] : phys_table_ids) {
    query_infos.emplace_back(InputTableInfo{
        db_id, table_id, executor->getDataMgr()->getTableMetadata(db_id, table_id)});
  }

  return query_infos;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<HashJoin> HashJoin::getSyntheticInstance(
    int db_id,
    std::string_view table1,
    std::string_view column1,
    std::string_view table2,
    std::string_view column2,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  auto a1 = getSyntheticColumnVar(db_id, table1, column1, 0, executor);
  auto a2 = getSyntheticColumnVar(db_id, table2, column2, 1, executor);

  auto qual_bin_oper = std::make_shared<hdk::ir::BinOper>(
      a1->ctx().boolean(), hdk::ir::OpType::kEq, hdk::ir::Qualifier::kOne, a1, a2);

  std::set<const hdk::ir::ColumnVar*> cvs =
      AllColumnVarsCollector::collect(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(data_provider, cvs, executor);

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          JoinType::INNER,
                                          preferred_hash_type,
                                          device_count,
                                          data_provider,
                                          column_cache,
                                          executor,
                                          {},
                                          {});
  return hash_table;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<HashJoin> HashJoin::getSyntheticInstance(
    const std::shared_ptr<const hdk::ir::BinOper> qual_bin_oper,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  std::set<const hdk::ir::ColumnVar*> cvs =
      AllColumnVarsCollector::collect(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(data_provider, cvs, executor);

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          JoinType::INNER,
                                          preferred_hash_type,
                                          device_count,
                                          data_provider,
                                          column_cache,
                                          executor,
                                          {},
                                          {});
  return hash_table;
}

std::pair<std::string, std::shared_ptr<HashJoin>> HashJoin::getSyntheticInstance(
    std::vector<std::shared_ptr<const hdk::ir::BinOper>> qual_bin_opers,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    DataProvider* data_provider,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  std::set<const hdk::ir::ColumnVar*> cvs;
  for (auto& qual : qual_bin_opers) {
    auto cv = AllColumnVarsCollector::collect(qual.get());
    cvs.insert(cv.begin(), cv.end());
  }
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(data_provider, cvs, executor);
  std::shared_ptr<HashJoin> hash_table;
  std::string error_msg;
  for (auto& qual : qual_bin_opers) {
    try {
      auto candidate_hash_table = HashJoin::getInstance(qual,
                                                        query_infos,
                                                        memory_level,
                                                        JoinType::INNER,
                                                        preferred_hash_type,
                                                        device_count,
                                                        data_provider,
                                                        column_cache,
                                                        executor,
                                                        {},
                                                        {});
      if (candidate_hash_table) {
        hash_table = candidate_hash_table;
      }
    } catch (HashJoinFail& e) {
      error_msg = e.what();
      continue;
    }
  }
  return std::make_pair(error_msg, hash_table);
}

InnerOuter HashJoin::normalizeColumnPair(const hdk::ir::Expr* lhs,
                                         const hdk::ir::Expr* rhs,
                                         SchemaProviderPtr schema_provider,
                                         const TemporaryTables* temporary_tables) {
  auto lhs_type = lhs->type();
  auto rhs_type = rhs->type();
  if (lhs_type->id() != rhs_type->id() || lhs_type->size() != rhs_type->size()) {
    throw HashJoinFail("Equijoin types must be identical, found: " +
                       lhs_type->toString() + ", " + rhs_type->toString());
  }
  if (!lhs_type->isInteger() && !lhs_type->isDateTime() && !lhs_type->isString() &&
      !lhs_type->isExtDictionary() && !lhs_type->isDecimal()) {
    throw HashJoinFail("Cannot apply hash join to inner column type " +
                       lhs_type->toString());
  }
  // Decimal types should be identical.
  if (lhs_type->isDecimal() && (lhs_type->as<hdk::ir::DecimalType>()->scale() !=
                                    rhs_type->as<hdk::ir::DecimalType>()->scale() ||
                                lhs_type->as<hdk::ir::DecimalType>()->precision() !=
                                    rhs_type->as<hdk::ir::DecimalType>()->precision())) {
    throw HashJoinFail("Equijoin with different decimal types");
  }

  const auto lhs_cast = dynamic_cast<const hdk::ir::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const hdk::ir::UOper*>(rhs);
  if ((lhs_type->isString() || lhs_type->isExtDictionary()) &&
      (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) ||
       (lhs_cast && !lhs_cast->isCast()) || (rhs_cast && !rhs_cast->isCast()))) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  // Casts to decimal are not suported.
  if (lhs_type->isDecimal() && (lhs_cast || rhs_cast)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const auto lhs_col = lhs_cast
                           ? dynamic_cast<const hdk::ir::ColumnVar*>(lhs_cast->operand())
                           : dynamic_cast<const hdk::ir::ColumnVar*>(lhs);
  const auto rhs_col = rhs_cast
                           ? dynamic_cast<const hdk::ir::ColumnVar*>(rhs_cast->operand())
                           : dynamic_cast<const hdk::ir::ColumnVar*>(rhs);
  if (!lhs_col && !rhs_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const hdk::ir::ColumnVar* inner_col{nullptr};
  const hdk::ir::ColumnVar* outer_col{nullptr};
  auto outer_type = lhs_type;
  auto inner_type = rhs_type;
  const hdk::ir::Expr* outer_expr{lhs};
  if (!lhs_col || (rhs_col && lhs_col->rteIdx() < rhs_col->rteIdx())) {
    inner_col = rhs_col;
    outer_col = lhs_col;
  } else {
    if (lhs_col && lhs_col->rteIdx() == 0) {
      throw HashJoinFail("Cannot use hash join for given expression");
    }
    inner_col = lhs_col;
    outer_col = rhs_col;
    std::swap(outer_type, inner_type);
    outer_expr = rhs;
  }
  if (!inner_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  if (!outer_col) {
    // check whether outer_col is a constant, i.e., inner_col = K;
    const auto outer_constant_col = dynamic_cast<const hdk::ir::Constant*>(outer_expr);
    if (outer_constant_col) {
      throw HashJoinFail(
          "Cannot use hash join for given expression: try to join with a constant "
          "value");
    }
    int outer_rte_idx = MaxRangeTableIndexCollector::collect(outer_expr);
    // The inner column candidate is not actually inner; the outer
    // expression contains columns which are at least as deep.
    if (inner_col->rteIdx() <= outer_rte_idx) {
      throw HashJoinFail("Cannot use hash join for given expression");
    }
  }
  // We need to fetch the actual type information from the schema provider since
  // Analyzer always reports nullable as true for inner table columns in left joins.
  const auto inner_col_info = schema_provider->getColumnInfo(*inner_col->columnInfo());
  CHECK(inner_col_info);
  const auto inner_col_real_type = inner_col_info->type;
  auto outer_col_type = !(dynamic_cast<const hdk::ir::FunctionOper*>(lhs)) && outer_col
                            ? outer_col->type()
                            : outer_type;
  // Casts from decimal are not supported.
  if ((inner_col_real_type->isDecimal() || outer_col_type->isDecimal()) &&
      (lhs_cast || rhs_cast)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }

  if (!(inner_col_real_type->isInteger() || inner_col_real_type->isDateTime() ||
        inner_col_real_type->isDecimal() || inner_col_real_type->isExtDictionary())) {
    throw HashJoinFail(
        "Can only apply hash join to integer-like types and dictionary encoded "
        "strings");
  }

  auto normalized_inner_col = inner_col;
  auto normalized_outer_col = outer_col ? outer_col : outer_expr;

  auto normalized_inner_type = normalized_inner_col->type();
  auto normalized_outer_type = normalized_outer_col->type();

  if (normalized_inner_type->isExtDictionary() !=
      normalized_outer_type->isExtDictionary()) {
    throw HashJoinFail(std::string("Could not build hash tables for incompatible types " +
                                   normalized_inner_type->toString() + " and " +
                                   normalized_outer_type->toString()));
  }

  return {normalized_inner_col, normalized_outer_col};
}

std::vector<InnerOuter> HashJoin::normalizeColumnPairs(
    const hdk::ir::BinOper* condition,
    SchemaProviderPtr schema_provider,
    const TemporaryTables* temporary_tables) {
  std::vector<InnerOuter> result;
  const auto lhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(condition->leftOperand());
  const auto rhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(condition->rightOperand());

  CHECK_EQ(static_cast<bool>(lhs_tuple_expr), static_cast<bool>(rhs_tuple_expr));
  if (lhs_tuple_expr) {
    const auto& lhs_tuple = lhs_tuple_expr->tuple();
    const auto& rhs_tuple = rhs_tuple_expr->tuple();
    CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
    for (size_t i = 0; i < lhs_tuple.size(); ++i) {
      result.push_back(normalizeColumnPair(
          lhs_tuple[i].get(), rhs_tuple[i].get(), schema_provider, temporary_tables));
    }
  } else {
    CHECK(!lhs_tuple_expr && !rhs_tuple_expr);
    result.push_back(normalizeColumnPair(condition->leftOperand(),
                                         condition->rightOperand(),
                                         schema_provider,
                                         temporary_tables));
  }

  return result;
}
