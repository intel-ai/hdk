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

#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"

#include <atomic>
#include <future>
#include <numeric>
#include <thread>

#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/Builders/PerfectHashTableBuilder.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/RuntimeFunctions.h"

// let's only consider CPU hahstable recycler at this moment
std::unique_ptr<HashtableRecycler> PerfectJoinHashTable::hash_table_cache_;
std::unique_ptr<HashingSchemeRecycler> PerfectJoinHashTable::hash_table_layout_cache_;
std::once_flag PerfectJoinHashTable::init_caches_flag_;

namespace {

InnerOuter get_cols(const hdk::ir::BinOper* qual_bin_oper,
                    SchemaProviderPtr schema_provider,
                    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->leftOperand();
  const auto rhs = qual_bin_oper->rightOperand();
  return HashJoin::normalizeColumnPair(lhs, rhs, schema_provider, temporary_tables);
}

HashEntryInfo get_bucketized_hash_entry_info(const hdk::ir::Type* context_type,
                                             ExpressionRange const& col_range,
                                             bool const is_bw_eq) {
  using EmptyRangeSize = boost::optional<size_t>;
  auto empty_range_check = [](ExpressionRange const& col_range,
                              bool const is_bw_eq) -> EmptyRangeSize {
    if (col_range.getIntMin() > col_range.getIntMax()) {
      CHECK_EQ(col_range.getIntMin(), int64_t(0));
      CHECK_EQ(col_range.getIntMax(), int64_t(-1));
      if (is_bw_eq) {
        return size_t(1);
      }
      return size_t(0);
    }
    return EmptyRangeSize{};
  };

  auto empty_range = empty_range_check(col_range, is_bw_eq);
  if (empty_range) {
    return {size_t(*empty_range), 1};
  }

  // size_t is not big enough for maximum possible range.
  if (col_range.getIntMin() == std::numeric_limits<int64_t>::min() &&
      col_range.getIntMax() == std::numeric_limits<int64_t>::max()) {
    throw TooManyHashEntries();
  }

  int64_t bucket_normalization = context_type->isDate() ? col_range.getBucket() : 1;
  CHECK_GT(bucket_normalization, 0);
  return {size_t(col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0)),
          bucket_normalization};
}

}  // namespace

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<PerfectJoinHashTable> PerfectJoinHashTable::getInstance(
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
  initCaches(executor->getConfigPtr());

  CHECK(qual_bin_oper->isEquivalence());
  const auto cols = get_cols(
      qual_bin_oper.get(), executor->getSchemaProvider(), executor->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  auto type = inner_col->type();
  auto is_string_type = type->isString() || type->isExtDictionary();
  auto col_range =
      getExpressionRange(is_string_type ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail(
        "Could not compute range for the expressions involved in the equijoin");
  }
  const auto rhs_source_col_range =
      is_string_type ? getExpressionRange(inner_col, query_infos, executor) : col_range;
  if (is_string_type) {
    // The nullable info must be the same as the source column.
    if (rhs_source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail(
          "Could not compute range for the expressions involved in the equijoin");
    }
    if (rhs_source_col_range.getIntMin() > rhs_source_col_range.getIntMax()) {
      // If the inner column expression range is empty, use the inner col range
      CHECK_EQ(rhs_source_col_range.getIntMin(), int64_t(0));
      CHECK_EQ(rhs_source_col_range.getIntMax(), int64_t(-1));
      col_range = rhs_source_col_range;
    } else {
      col_range = ExpressionRange::makeIntRange(
          std::min(rhs_source_col_range.getIntMin(), col_range.getIntMin()),
          std::max(rhs_source_col_range.getIntMax(), col_range.getIntMax()),
          0,
          rhs_source_col_range.hasNulls());
    }
  }

  // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
  const auto max_hash_entry_count =
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? static_cast<size_t>(std::numeric_limits<int32_t>::max() / sizeof(int32_t))
          : static_cast<size_t>(std::numeric_limits<int32_t>::max());

  auto bucketized_entry_count_info =
      get_bucketized_hash_entry_info(type, col_range, qual_bin_oper->isBwEq());
  auto bucketized_entry_count = bucketized_entry_count_info.getNormalizedHashEntryCount();

  if (bucketized_entry_count > max_hash_entry_count) {
    throw TooManyHashEntries();
  }

  // We don't want to build huge and very sparse tables
  // to consume lots of memory.
  if (bucketized_entry_count > executor->getConfig().exec.join.huge_join_hash_threshold) {
    const auto& query_info = get_inner_query_info(inner_col->tableId(), query_infos).info;
    if (query_info.getNumTuplesUpperBound() * 100 <
        executor->getConfig().exec.join.huge_join_hash_min_load *
            bucketized_entry_count) {
      throw TooManyHashEntries();
    }
  }

  if (qual_bin_oper->isBwEq() &&
      col_range.getIntMax() >= std::numeric_limits<int64_t>::max()) {
    throw HashJoinFail("Cannot translate null value for kBW_EQ");
  }
  std::vector<InnerOuter> inner_outer_pairs;
  inner_outer_pairs.emplace_back(inner_col, cols.second);
  auto hashtable_cache_key =
      HashtableRecycler::getHashtableCacheKey(inner_outer_pairs,
                                              qual_bin_oper->opType(),
                                              join_type,
                                              hashtable_build_dag_map,
                                              executor);
  auto hash_key = hashtable_cache_key.first;
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  if (VLOGGING(1)) {
    ts1 = std::chrono::steady_clock::now();
  }
  auto join_hash_table = std::shared_ptr<PerfectJoinHashTable>(
      new PerfectJoinHashTable(qual_bin_oper,
                               inner_col,
                               query_infos,
                               memory_level,
                               join_type,
                               preferred_hash_type,
                               col_range,
                               rhs_source_col_range,
                               data_provider,
                               column_cache,
                               executor,
                               device_count,
                               hash_key,
                               hashtable_cache_key.second,
                               table_id_to_node_map));
  try {
    join_hash_table->reify();
  } catch (const TableMustBeReplicated& e) {
    // Throw a runtime error to abort the query
    join_hash_table->freeHashBufferMemory();
    throw std::runtime_error(e.what());
  } catch (const HashJoinFail& e) {
    // HashJoinFail exceptions log an error and trigger a retry with a join loop (if
    // possible)
    join_hash_table->freeHashBufferMemory();
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in equijoin | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(std::string("Could not build hash tables for equijoin | ") +
                       e.what());
  } catch (const OutOfMemory& e) {
    throw HashJoinFail(
        std::string("Ran out of memory while building hash tables for equijoin | ") +
        e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Fatal error while attempting to build hash tables for join: ") +
        e.what());
  }
  if (VLOGGING(1)) {
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "Built perfect hash table "
            << getHashTypeString(join_hash_table->getHashType()) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

void PerfectJoinHashTable::initCaches(ConfigPtr config) {
  std::call_once(init_caches_flag_, [&]() {
    hash_table_cache_ = std::make_unique<HashtableRecycler>(
        config, CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    hash_table_layout_cache_ = std::make_unique<HashingSchemeRecycler>(config);
  });
}

bool needs_dictionary_translation(const hdk::ir::ColumnVar* inner_col,
                                  const hdk::ir::Expr* outer_col_expr,
                                  const Executor* executor) {
  auto schema_provider = executor->getSchemaProvider();
  CHECK(schema_provider);
  const auto inner_col_info = schema_provider->getColumnInfo(*inner_col->columnInfo());
  const auto& inner_type = inner_col->type();
  // Only strings may need dictionary translation.
  if (!inner_type->isExtDictionary() && !inner_type->isString()) {
    return false;
  }
  const auto outer_col = dynamic_cast<const hdk::ir::ColumnVar*>(outer_col_expr);
  CHECK(outer_col);
  const auto outer_col_info = schema_provider->getColumnInfo(*outer_col->columnInfo());
  // Don't want to deal with temporary tables for now, require translation.
  if (!inner_col_info || !outer_col_info) {
    return true;
  }
  const auto& outer_type = outer_col->type();
  CHECK(outer_type->isExtDictionary() || outer_type->isString());
  // If the two columns don't share the dictionary, translation is needed.
  if (outer_type->isExtDictionary() != inner_type->isExtDictionary()) {
    return true;
  }
  CHECK(inner_type->isExtDictionary() && outer_type->isExtDictionary());

  auto inner_dict_id = inner_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  auto outer_dict_id = outer_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  const auto inner_str_dict_proxy =
      executor->getStringDictionaryProxy(inner_dict_id, true);
  CHECK(inner_str_dict_proxy);
  const auto outer_str_dict_proxy =
      executor->getStringDictionaryProxy(outer_dict_id, true);
  CHECK(outer_str_dict_proxy);

  return *inner_str_dict_proxy != *outer_str_dict_proxy;
}

bool PerfectJoinHashTable::isOneToOneHashPossible(
    const std::vector<ColumnsForDevice>& columns_per_device) const {
  CHECK(!inner_outer_pairs_.empty());
  auto rhs_col_type = inner_outer_pairs_.front().first->type();
  const auto max_unique_hash_input_entries =
      get_bucketized_hash_entry_info(
          rhs_col_type, rhs_source_col_range_, qual_bin_oper_->isBwEq())
          .getNormalizedHashEntryCount() +
      rhs_source_col_range_.hasNulls();
  for (const auto& device_columns : columns_per_device) {
    CHECK(!device_columns.join_columns.empty());
    const auto rhs_join_col_num_entries = device_columns.join_columns.front().num_elems;
    if (rhs_join_col_num_entries > max_unique_hash_input_entries) {
      VLOG(1) << "Skipping attempt to build perfect hash one-to-one table as number of "
                 "rhs column entries ("
              << rhs_join_col_num_entries << ") exceeds range for rhs join column ("
              << max_unique_hash_input_entries << ").";
      return false;
    }
  }
  return true;
}

void PerfectJoinHashTable::reify() {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0, device_count_);
  const auto cols = get_cols(
      qual_bin_oper_.get(), executor_->getSchemaProvider(), executor_->temporary_tables_);
  const auto inner_col = cols.first;
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return;
  }
  if (query_info.getNumTuplesUpperBound() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  std::vector<std::future<void>> init_threads;

  inner_outer_pairs_.push_back(cols);
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));

  std::vector<std::vector<FragmentInfo>> fragments_per_device;
  std::vector<ColumnsForDevice> columns_per_device;
  std::vector<std::unique_ptr<GpuAllocator>> dev_buff_owners;
  auto buffer_provider = executor_->getBufferProvider();
  if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      dev_buff_owners.emplace_back(
          std::make_unique<GpuAllocator>(buffer_provider, device_id));
    }
  }

  for (int device_id = 0; device_id < device_count_; ++device_id) {
    fragments_per_device.emplace_back(query_info.fragments);
    columns_per_device.emplace_back(
        fetchColumnsForDevice(fragments_per_device.back(),
                              device_id,
                              memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL
                                  ? dev_buff_owners[device_id].get()
                                  : nullptr));
  }
  // Now check if on the number of entries per column exceeds the rhs join hash table
  // range, and skip trying to build a One-to-One hash table if so
  if (!isOneToOneHashPossible(columns_per_device)) {
    hash_type_ = HashType::OneToMany;
  }

  try {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto chunk_key = genChunkKey(fragments_per_device[device_id],
                                         inner_outer_pairs_.front().second,
                                         inner_outer_pairs_.front().first);
      if (device_id == 0 && hashtable_cache_key_ == EMPTY_HASHED_PLAN_DAG_KEY &&
          getInnerTableId() > 0) {
        // sometimes we cannot retrieve query plan dag, so try to recycler cache
        // with the old-fashioned cache key if we deal with hashtable of non-temporary
        // table
        auto outer_col =
            dynamic_cast<const hdk::ir::ColumnVar*>(inner_outer_pairs_.front().second);
        AlternativeCacheKeyForPerfectHashJoin cache_key{
            col_range_,
            inner_col,
            outer_col ? outer_col : inner_col,
            chunk_key,
            columns_per_device[device_id].join_columns.front().num_elems,
            qual_bin_oper_->opType(),
            join_type_};
        hashtable_cache_key_ = getAlternativeCacheKey(cache_key);
        VLOG(2) << "Use alternative hashtable cache key due to unavailable query plan "
                   "dag extraction";
      }
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        chunk_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  } catch (const NeedsOneToManyHash& e) {
    VLOG(1) << "RHS/Inner hash join values detected to not be unique, falling back to "
               "One-to-Many hash layout.";
    CHECK(hash_type_ == HashType::OneToOne);
    hash_type_ = HashType::OneToMany;
    freeHashBufferMemory();
    init_threads.clear();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      CHECK_EQ(dev_buff_owners.size(), size_t(device_count_));
    }
    CHECK_EQ(columns_per_device.size(), size_t(device_count_));
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto chunk_key = genChunkKey(fragments_per_device[device_id],
                                         inner_outer_pairs_.front().second,
                                         inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        chunk_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  }
}

Data_Namespace::MemoryLevel PerfectJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(
            inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      needs_dict_translation_ = true;
      return Data_Namespace::CPU_LEVEL;
    }
  }
  return memory_level_;
}

ColumnsForDevice PerfectJoinHashTable::fetchColumnsForDevice(
    const std::vector<FragmentInfo>& fragments,
    const int device_id,
    DeviceAllocator* dev_buff_owner) {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<std::shared_ptr<void>> malloc_owner;
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    if (inner_col->isVirtual()) {
      if (inner_outer_pairs_.size() != 1) {
        throw HashJoinFail("Only single predicate row-id joins are currently supported.");
      }
      // push back an empty join column, which will prevent hash table build. the probe
      // for a row-id hash join simply does null checks and returns the input row-id
      auto type = inner_col->type();
      join_columns.emplace_back(
          JoinColumn{/*ptr=*/nullptr,
                     /*col_chunks_buff_sz=*/0,
                     /*num_chunks=*/1,
                     /*num_elems=*/0,
                     /*elem_sz=*/static_cast<size_t>(type->size())});
      continue;
    }
    join_columns.emplace_back(fetchJoinColumn(inner_col,
                                              fragments,
                                              effective_memory_level,
                                              device_id,
                                              chunks_owner,
                                              dev_buff_owner,
                                              malloc_owner,
                                              executor_,
                                              &column_cache_));
    auto type = inner_col->type();
    join_column_types.emplace_back(
        JoinColumnTypeInfo{static_cast<size_t>(type->size()),
                           0,
                           0,
                           inline_fixed_encoding_null_value(type),
                           isBitwiseEq(),
                           0,
                           get_join_column_type_kind(type)});
  }
  return {join_columns, join_column_types, chunks_owner, malloc_owner};
}

void PerfectJoinHashTable::reifyForDevice(const ChunkKey& chunk_key,
                                          const ColumnsForDevice& columns_for_device,
                                          const HashType layout,
                                          const int device_id,
                                          const logger::ThreadId parent_thread_id) {
  DEBUG_TIMER_NEW_THREAD(parent_thread_id);
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  CHECK_EQ(columns_for_device.join_columns.size(), size_t(1));
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  auto& join_column = columns_for_device.join_columns.front();
  if (layout == HashType::OneToOne) {
    const auto err = initHashTableForDevice(chunk_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            layout,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw NeedsOneToManyHash();
    }
  } else {
    const auto err = initHashTableForDevice(chunk_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            HashType::OneToMany,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw std::runtime_error("Unexpected error building one to many hash table: " +
                               std::to_string(err));
    }
  }
}

int PerfectJoinHashTable::initHashTableForDevice(
    const ChunkKey& chunk_key,
    const JoinColumn& join_column,
    const InnerOuter& cols,
    const HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto timer = DEBUG_TIMER(__func__);
  const auto inner_col = cols.first;
  CHECK(inner_col);

  auto hash_entry_info =
      get_bucketized_hash_entry_info(inner_col->type(), col_range_, isBitwiseEq());
  if (!hash_entry_info && layout == HashType::OneToOne) {
    // TODO: what is this for?
    return 0;
  }

#ifndef HAVE_CUDA
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  int err{0};
  const int32_t hash_join_invalid_val{-1};
  auto hashtable_layout = layout;
  auto allow_hashtable_recycling = HashtableRecycler::isSafeToCacheHashtable(
      table_id_to_node_map_, needs_dict_translation_, inner_col->tableId());
  if (allow_hashtable_recycling) {
    auto cached_hashtable_layout_type = hash_table_layout_cache_->getItemFromCache(
        hashtable_cache_key_,
        CacheItemType::HT_HASHING_SCHEME,
        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
        {});
    if (cached_hashtable_layout_type) {
      hash_type_ = *cached_hashtable_layout_type;
      hashtable_layout = hash_type_;
      VLOG(1) << "Recycle hashtable layout: " << getHashTypeString(hashtable_layout);
    }
  }
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    CHECK(!chunk_key.empty());
    std::shared_ptr<PerfectHashTable> hash_table{nullptr};
    if (allow_hashtable_recycling) {
      hash_table = initHashTableOnCpuFromCache(hashtable_cache_key_,
                                               CacheItemType::PERFECT_HT,
                                               DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    }
    if (!hash_table) {
      std::unique_lock<std::mutex> str_proxy_translation_lock(
          str_proxy_translation_mutex_);
      // It's not ideal to populate the str dict proxy translation map at the per device
      // init func, but currently with the hash table cache lookup (above) at this
      // level, if we do the translation in PerfectJoinHashTable::reify, we don't know
      // if the hash table is cached and so needlessly compute a potentially expensive
      // proxy translation even if we have the hash table already cached.
      // Todo(todd/yoonmin): Hoist cache lookup to PerfectJoinHashTable::reify and then
      // move this proxy translation to that level as well, conditioned on the hash
      // table not being cached.
      if (needs_dict_translation_ && !str_proxy_translation_map_) {
        CHECK_GE(inner_outer_pairs_.size(), 1UL);
        str_proxy_translation_map_ = HashJoin::translateInnerToOuterStrDictProxies(
            inner_outer_pairs_.front(), executor_);
      }
    }
    decltype(std::chrono::steady_clock::now()) ts1, ts2;
    ts1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      if (!hash_table) {
        // Try to get hash table from cache again, since if we are building
        // for multiple devices, all devices except the first to take
        // cpu_hash_table_buff_lock should find their hash table cached now
        // from the first device to run
        hash_table = initHashTableOnCpuFromCache(hashtable_cache_key_,
                                                 CacheItemType::PERFECT_HT,
                                                 DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
        if (!hash_table) {
          PerfectJoinHashTableBuilder builder;
          if (hashtable_layout == HashType::OneToOne) {
            builder.initOneToOneHashTableOnCpu(join_column,
                                               col_range_,
                                               isBitwiseEq(),
                                               cols,
                                               str_proxy_translation_map_,
                                               join_type_,
                                               hashtable_layout,
                                               hash_entry_info,
                                               hash_join_invalid_val,
                                               executor_);
            hash_table = builder.getHashTable();
          } else {
            builder.initOneToManyHashTableOnCpu(join_column,
                                                col_range_,
                                                isBitwiseEq(),
                                                cols,
                                                str_proxy_translation_map_,
                                                hash_entry_info,
                                                hash_join_invalid_val,
                                                executor_);
            hash_table = builder.getHashTable();
          }
          ts2 = std::chrono::steady_clock::now();
          auto build_time =
              std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count();
          if (allow_hashtable_recycling && hash_table) {
            // add ht-related items to cache iff we have a valid hashtable
            hash_table_layout_cache_->putItemToCache(
                hashtable_cache_key_,
                hashtable_layout,
                CacheItemType::HT_HASHING_SCHEME,
                DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                0,
                0,
                {});
            putHashTableOnCpuToCache(hashtable_cache_key_,
                                     CacheItemType::PERFECT_HT,
                                     hash_table,
                                     DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                     build_time);
          }
        }
      }
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      auto buffer_provider = executor_->getBufferProvider();
      auto type = inner_col->type();
      CHECK(type->isString() || type->isExtDictionary());
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);

      PerfectJoinHashTableBuilder gpu_builder;
      gpu_builder.allocateDeviceMemory(join_column,
                                       hash_table->getLayout(),
                                       hash_entry_info,
                                       device_id,
                                       device_count_,
                                       executor_);
      std::shared_ptr<PerfectHashTable> gpu_hash_table = gpu_builder.getHashTable();
      CHECK(gpu_hash_table);
      auto gpu_buffer_ptr = gpu_hash_table->getGpuBuffer();
      CHECK(gpu_buffer_ptr);

      CHECK(hash_table);
      // GPU size returns reserved size
      CHECK_LE(hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
               gpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU));
      buffer_provider->copyToDevice(
          gpu_buffer_ptr,
          hash_table->getCpuBuffer(),
          hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
          device_id);
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = std::move(gpu_hash_table);
#else
      UNREACHABLE();
#endif
    } else {
      CHECK(hash_table);
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = hash_table;
    }
  } else {
#ifdef HAVE_CUDA
    PerfectJoinHashTableBuilder builder;
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    builder.allocateDeviceMemory(join_column,
                                 hashtable_layout,
                                 hash_entry_info,
                                 device_id,
                                 device_count_,
                                 executor_);
    builder.initHashTableOnGpu(chunk_key,
                               join_column,
                               col_range_,
                               isBitwiseEq(),
                               cols,
                               join_type_,
                               hashtable_layout,
                               hash_entry_info,
                               hash_join_invalid_val,
                               device_id,
                               device_count_,
                               executor_);
    CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
    hash_tables_for_device_[device_id] = builder.getHashTable();
    if (!err && allow_hashtable_recycling && hash_tables_for_device_[device_id]) {
      // add layout to cache iff we have a valid hashtable
      hash_table_layout_cache_->putItemToCache(
          hashtable_cache_key_,
          hash_tables_for_device_[device_id]->getLayout(),
          CacheItemType::HT_HASHING_SCHEME,
          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
          0,
          0,
          {});
    }
#else
    UNREACHABLE();
#endif
  }
  return err;
}

ChunkKey PerfectJoinHashTable::genChunkKey(const std::vector<FragmentInfo>& fragments,
                                           const hdk::ir::Expr* outer_col_expr,
                                           const hdk::ir::ColumnVar* inner_col) const {
  ChunkKey chunk_key{inner_col->dbId(), inner_col->tableId(), inner_col->columnId()};
  auto type = inner_col->type();
  if (type->isExtDictionary()) {
    size_t outer_elem_count = 0;
    const auto outer_col = dynamic_cast<const hdk::ir::ColumnVar*>(outer_col_expr);
    CHECK(outer_col);
    const auto& outer_query_info = getInnerQueryInfo(outer_col).info;
    for (auto& frag : outer_query_info.fragments) {
      outer_elem_count = frag.getNumTuples();
    }
    chunk_key.push_back(outer_elem_count);
  }
  if (fragments.size() < 2) {
    chunk_key.push_back(fragments.front().fragmentId);
  }
  return chunk_key;
}

std::shared_ptr<PerfectHashTable> PerfectJoinHashTable::initHashTableOnCpuFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  CHECK(hash_table_cache_);
  auto timer = DEBUG_TIMER(__func__);
  auto hashtable_ptr =
      hash_table_cache_->getItemFromCache(key, item_type, device_identifier);
  if (hashtable_ptr) {
    return std::dynamic_pointer_cast<PerfectHashTable>(hashtable_ptr);
  }
  return nullptr;
}

void PerfectJoinHashTable::putHashTableOnCpuToCache(
    QueryPlanHash key,
    CacheItemType item_type,
    std::shared_ptr<PerfectHashTable> hashtable_ptr,
    DeviceIdentifier device_identifier,
    size_t hashtable_building_time) {
  CHECK(hash_table_cache_);
  CHECK(hashtable_ptr && !hashtable_ptr->getGpuBuffer());
  hash_table_cache_->putItemToCache(
      key,
      hashtable_ptr,
      item_type,
      device_identifier,
      hashtable_ptr->getHashTableBufferSize(ExecutorDeviceType::CPU),
      hashtable_building_time);
}

llvm::Value* PerfectJoinHashTable::codegenHashTableLoad(const size_t table_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto hash_ptr = HashJoin::codegenHashTableLoad(table_idx, executor_);
  if (hash_ptr->getType()->isIntegerTy(64)) {
    return hash_ptr;
  }
  CHECK(hash_ptr->getType()->isPointerTy());
  return executor_->cgen_state_->ir_builder_.CreatePtrToInt(
      get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables"),
      llvm::Type::getInt64Ty(executor_->cgen_state_->context_));
}

std::vector<llvm::Value*> PerfectJoinHashTable::getHashJoinArgs(
    llvm::Value* hash_ptr,
    const hdk::ir::Expr* key_col,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CodeGenerator code_generator(executor_);
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto key_col_type = key_col->type();
  auto hash_entry_info =
      get_bucketized_hash_entry_info(key_col_type, col_range_, isBitwiseEq());

  std::vector<llvm::Value*> hash_join_idx_args{
      hash_ptr,
      executor_->cgen_state_->castToTypeIn(key_lvs.front(), 64),
      executor_->cgen_state_->llInt(col_range_.getIntMin()),
      executor_->cgen_state_->llInt(col_range_.getIntMax())};
  auto key_col_logical_type = key_col->type()->canonicalize();
  if (key_col_logical_type->nullable() || isBitwiseEq()) {
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
        inline_fixed_encoding_null_value(key_col_logical_type)));
  }
  auto special_date_bucketization_case = key_col_type->isDate();
  if (isBitwiseEq()) {
    if (special_date_bucketization_case) {
      hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
          col_range_.getIntMax() / hash_entry_info.bucket_normalization + 1));
    } else {
      hash_join_idx_args.push_back(
          executor_->cgen_state_->llInt(col_range_.getIntMax() + 1));
    }
  }

  if (special_date_bucketization_case) {
    hash_join_idx_args.emplace_back(
        executor_->cgen_state_->llInt(hash_entry_info.bucket_normalization));
  }

  return hash_join_idx_args;
}

HashJoinMatchingSet PerfectJoinHashTable::codegenMatchingSet(const CompilationOptions& co,
                                                             const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto cols = get_cols(
      qual_bin_oper_.get(), executor_->getSchemaProvider(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  auto pos_ptr = codegenHashTableLoad(index);
  CHECK(pos_ptr);
  const auto key_col_var = dynamic_cast<const hdk::ir::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const hdk::ir::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  auto hash_join_idx_args = getHashJoinArgs(pos_ptr, key_col, co);
  const int64_t sub_buff_size = getComponentBufferSize();
  auto key_col_type = key_col->type();

  auto bucketize = key_col_type->isDate();
  return HashJoin::codegenMatchingSet(hash_join_idx_args,
                                      key_col_type->nullable(),
                                      isBitwiseEq(),
                                      sub_buff_size,
                                      executor_,
                                      bucketize);
}

size_t PerfectJoinHashTable::offsetBufferOff() const noexcept {
  return 0;
}

size_t PerfectJoinHashTable::countBufferOff() const noexcept {
  return getComponentBufferSize();
}

size_t PerfectJoinHashTable::payloadBufferOff() const noexcept {
  return 2 * getComponentBufferSize();
}

size_t PerfectJoinHashTable::getComponentBufferSize() const noexcept {
  if (hash_tables_for_device_.empty()) {
    return 0;
  }
  auto hash_table = hash_tables_for_device_.front();
  if (hash_table && hash_table->getLayout() == HashType::OneToMany) {
    return hash_table->getEntryCount() * sizeof(int32_t);
  } else {
    return 0;
  }
}

HashTable* PerfectJoinHashTable::getHashTableForDevice(const size_t device_id) const {
  CHECK_LT(device_id, hash_tables_for_device_.size());
  return hash_tables_for_device_[device_id].get();
}

std::string PerfectJoinHashTable::toString(const ExecutorDeviceType device_type,
                                           const int device_id,
                                           bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  auto buffer_provider = executor_->getBufferProvider();
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    buffer_provider->copyFromDevice(buffer_copy.get(),
                                    reinterpret_cast<const int8_t*>(buffer),
                                    buffer_size,
                                    device_id);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return HashTable::toString("perfect",
                             getHashTypeString(hash_type_),
                             0,
                             0,
                             hash_table ? hash_table->getEntryCount() : 0,
                             ptr1,
                             ptr2,
                             ptr3,
                             ptr4,
                             buffer_size,
                             raw);
}

std::set<DecodedJoinHashBufferEntry> PerfectJoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  auto buffer_provider = executor_->getBufferProvider();
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    buffer_provider->copyFromDevice(buffer_copy.get(),
                                    reinterpret_cast<const int8_t*>(buffer),
                                    buffer_size,
                                    device_id);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return HashTable::toSet(0,
                          0,
                          hash_table ? hash_table->getEntryCount() : 0,
                          ptr1,
                          ptr2,
                          ptr3,
                          ptr4,
                          buffer_size);
}

llvm::Value* PerfectJoinHashTable::codegenSlot(const CompilationOptions& co,
                                               const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  using namespace std::string_literals;

  CHECK(getHashType() == HashType::OneToOne);
  const auto cols = get_cols(
      qual_bin_oper_.get(), executor_->getSchemaProvider(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  CodeGenerator code_generator(executor_);
  const auto key_col_var = dynamic_cast<const hdk::ir::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const hdk::ir::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto key_col_type = key_col->type();

  if (val_col->isVirtual()) {
    // short-circuit hash table load. we know the key will be the row number we want to
    // access, so just return the key
    CHECK(val_col_var);  // make sure this virtual column is a column

    std::vector<llvm::Value*> hash_join_idx_args{
        executor_->cgen_state_->castToTypeIn(key_lvs.front(), 64)};
    hash_join_idx_args.push_back(
        executor_->cgen_state_->llInt(rhs_source_col_range_.getIntMin()));
    hash_join_idx_args.push_back(
        executor_->cgen_state_->llInt(rhs_source_col_range_.getIntMax()));

    auto key_col_logical_type = key_col->type()->canonicalize();
    if (key_col_logical_type->nullable() || isBitwiseEq()) {
      hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
          inline_fixed_encoding_null_value(key_col_logical_type)));
    }

    std::string fname{"rowid_hash_join_idx"};
    if (key_col_type->nullable()) {
      fname += "_nullable";
    }
    return executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
  }
  auto hash_ptr = codegenHashTableLoad(index);
  CHECK(hash_ptr);
  const auto hash_join_idx_args = getHashJoinArgs(hash_ptr, key_col, co);

  std::string fname(key_col_type->isDate() ? "bucketized_hash_join_idx"s
                                           : "hash_join_idx"s);

  if (isBitwiseEq()) {
    fname += "_bitwise";
  }

  if (!isBitwiseEq() && key_col_type->nullable()) {
    fname += "_nullable";
  }
  return executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
}

const InputTableInfo& PerfectJoinHashTable::getInnerQueryInfo(
    const hdk::ir::ColumnVar* inner_col) const {
  return get_inner_query_info(inner_col->tableId(), query_infos_);
}

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos) {
  std::optional<size_t> ti_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (inner_table_id == query_infos[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK(ti_idx);
  return query_infos[*ti_idx];
}

bool PerfectJoinHashTable::isBitwiseEq() const {
  return qual_bin_oper_->isBwEq();
}
