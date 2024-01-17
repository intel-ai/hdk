/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "QueryExecutionContext.h"
#include "AggregateUtils.h"
#include "CompilationOptions.h"
#include "DeviceKernel.h"
#include "Execute.h"
#include "GpuInitGroups.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "QueryMemoryInitializer.h"
#include "RelAlgExecutionUnit.h"
#include "ResultSetReduction.h"
#include "SpeculativeTopN.h"
#include "StreamingTopN.h"

#include "ResultSet/QueryMemoryDescriptor.h"
#include "ResultSet/ResultSet.h"
#include "Shared/likely.h"

QueryExecutionContext::QueryExecutionContext(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    Executor* executor,
    const ExecutorDeviceType device_type,
    const ExecutorDispatchMode dispatch_mode,
    bool use_groupby_buffer_desc,
    const int device_id,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    const size_t thread_idx)
    : query_mem_desc_(query_mem_desc)
    , executor_(executor)
    , device_type_(device_type)
    , dispatch_mode_(dispatch_mode)
    , row_set_mem_owner_(row_set_mem_owner)
    , output_columnar_(output_columnar) {
  CHECK(executor);
  if (device_type == ExecutorDeviceType::GPU) {
    gpu_allocator_ =
        std::make_unique<GpuAllocator>(executor->getBufferProvider(), device_id);
  }

  query_buffers_ = std::make_unique<QueryMemoryInitializer>(ra_exe_unit,
                                                            query_mem_desc,
                                                            device_id,
                                                            device_type,
                                                            dispatch_mode,
                                                            output_columnar,
                                                            sort_on_gpu,
                                                            use_groupby_buffer_desc,
                                                            num_rows,
                                                            col_buffers,
                                                            frag_offsets,
                                                            row_set_mem_owner,
                                                            gpu_allocator_.get(),
                                                            thread_idx,
                                                            executor);
}

std::unique_ptr<QueryExecutionContext> QueryExecutionContext::create(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    Executor* executor,
    const ExecutorDeviceType device_type,
    const ExecutorDispatchMode dispatch_mode,
    const bool use_groupby_buffer_desc,
    const int device_id,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    const size_t thread_idx) {
  auto timer = DEBUG_TIMER(__func__);
  if (frag_offsets.empty()) {
    return nullptr;
  }
  return std::unique_ptr<QueryExecutionContext>(
      new QueryExecutionContext(ra_exe_unit,
                                query_mem_desc,
                                executor,
                                device_type,
                                dispatch_mode,
                                use_groupby_buffer_desc,
                                device_id,
                                num_rows,
                                col_buffers,
                                frag_offsets,
                                row_set_mem_owner,
                                output_columnar,
                                sort_on_gpu,
                                thread_idx));
}

ResultSetPtr QueryExecutionContext::groupBufferToDeinterleavedResults(
    const size_t i) const {
  CHECK(!output_columnar_);
  const auto& result_set = query_buffers_->getResultSet(i);
  auto deinterleaved_query_mem_desc =
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  deinterleaved_query_mem_desc.setHasInterleavedBinsOnGpu(false);
  deinterleaved_query_mem_desc.useConsistentSlotWidthSize(8);

  auto deinterleaved_result_set =
      std::make_shared<ResultSet>(result_set->getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  ExecutorDeviceType::CPU,
                                  -1,
                                  deinterleaved_query_mem_desc,
                                  row_set_mem_owner_,
                                  executor_->getDataMgr(),
                                  executor_->blockSize(),
                                  executor_->gridSize());
  auto deinterleaved_storage =
      deinterleaved_result_set->allocateStorage(executor_->plan_state_->init_agg_vals_);
  auto deinterleaved_buffer =
      reinterpret_cast<int64_t*>(deinterleaved_storage->getUnderlyingBuffer());
  const auto rows_ptr = result_set->getStorage()->getUnderlyingBuffer();
  size_t deinterleaved_buffer_idx = 0;
  const size_t agg_col_count{query_mem_desc_.getSlotCount()};
  auto do_work = [&](const size_t bin_base_off) {
    std::vector<int64_t> agg_vals(agg_col_count, 0);
    memcpy(&agg_vals[0],
           &executor_->plan_state_->init_agg_vals_[0],
           agg_col_count * sizeof(agg_vals[0]));
    ResultSetReduction::reduceSingleRow(rows_ptr + bin_base_off,
                                        executor_->warpSize(),
                                        false,
                                        true,
                                        agg_vals,
                                        query_mem_desc_,
                                        result_set->getTargetInfos(),
                                        executor_->plan_state_->init_agg_vals_);
    for (size_t agg_idx = 0; agg_idx < agg_col_count;
         ++agg_idx, ++deinterleaved_buffer_idx) {
      deinterleaved_buffer[deinterleaved_buffer_idx] = agg_vals[agg_idx];
    }
  };
  if (executor_->getConfig().exec.interrupt.enable_non_kernel_time_query_interrupt) {
    for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0), bin_idx = 0;
         bin_idx < result_set->entryCount();
         ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      if (UNLIKELY((bin_idx & 0xFFFF) == 0 &&
                   executor_->checkNonKernelTimeInterrupted())) {
        throw std::runtime_error(
            "Query execution has interrupted during result set reduction");
      }
      do_work(bin_base_off);
    }
  } else {
    for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0), bin_idx = 0;
         bin_idx < result_set->entryCount();
         ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      do_work(bin_base_off);
    }
  }
  query_buffers_->resetResultSet(i);
  return deinterleaved_result_set;
}

int64_t QueryExecutionContext::getAggInitValForIndex(const size_t index) const {
  CHECK(query_buffers_);
  return query_buffers_->getAggInitValForIndex(index);
}

ResultSetPtr QueryExecutionContext::getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const CompilationOptions& co) const {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> results_per_sm;
  CHECK(query_buffers_);
  const auto group_by_buffers_size = query_buffers_->getNumBuffers();
  if (device_type_ == ExecutorDeviceType::CPU) {
    const size_t expected_num_buffers = query_mem_desc.hasVarlenOutput() ? 2 : 1;
    CHECK_EQ(expected_num_buffers, group_by_buffers_size);
    return groupBufferToResults(0);
  }
  const size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  const size_t group_by_output_buffers_size =
      group_by_buffers_size - (query_mem_desc.hasVarlenOutput() ? 1 : 0);
  for (size_t i = 0; i < group_by_output_buffers_size; i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i), std::vector<size_t>{});
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(
      ra_exe_unit, results_per_sm, row_set_mem_owner_, query_mem_desc, co);
}

ResultSetPtr QueryExecutionContext::groupBufferToResults(const size_t i) const {
  if (query_mem_desc_.interleavedBins(device_type_)) {
    return groupBufferToDeinterleavedResults(i);
  }
  return query_buffers_->getResultSetOwned(i);
}

namespace {

int32_t aggregate_error_codes(const std::vector<int32_t>& error_codes) {
  // Check overflow / division by zero / interrupt first
  for (const auto err : error_codes) {
    if (err > 0) {
      return err;
    }
  }
  for (const auto err : error_codes) {
    if (err) {
      return err;
    }
  }
  return 0;
}

}  // namespace

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationContext* compilation_context,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    Data_Namespace::DataMgr* data_mgr,
    BufferProvider* buffer_provider,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const size_t shared_memory_size,
    int32_t* error_code,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    const std::vector<int64_t>& join_hash_tables) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(lauchGpuCode);
  CHECK(gpu_allocator_);
  CHECK(query_buffers_);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  bool is_group_by{query_mem_desc_.isGroupBy()};

  CHECK(compilation_context);
  auto kernel = create_device_kernel(
      compilation_context, data_mgr->getGpuMgr()->getPlatform(), device_id);

  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(grid_size_x * block_size_x);

  auto prepareClock = kernel->make_clock();
  auto launchClock = kernel->make_clock();
  auto finishClock = kernel->make_clock();

  if (executor_->getConfig().exec.watchdog.enable_dynamic || (allow_runtime_interrupt)) {
    prepareClock->start();
  }

  if (executor_->getConfig().exec.watchdog.enable_dynamic) {
    kernel->initializeDynamicWatchdog(
        executor_->interrupted_.load(),
        executor_->deviceCycles(executor_->getConfig().exec.watchdog.time_limit),
        executor_->getConfig().exec.watchdog.time_limit);
  }

  if (allow_runtime_interrupt) {
    kernel->initializeRuntimeInterrupter();
  }

  auto [kernel_params, kernel_metadata_gpu_buf] = prepareKernelParams(col_buffers,
                                                                      literal_buff,
                                                                      num_rows,
                                                                      frag_offsets,
                                                                      scan_limit,
                                                                      init_agg_vals,
                                                                      error_codes,
                                                                      num_tables,
                                                                      join_hash_tables,
                                                                      buffer_provider,
                                                                      device_id,
                                                                      hoist_literals,
                                                                      is_group_by);

  CHECK_EQ(static_cast<size_t>(KERN_PARAM_COUNT), kernel_params.size());

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  const auto total_thread_count = block_size_x * grid_size_x;
  const auto err_desc = kernel_params[ERROR_CODE];

  if (is_group_by) {
    CHECK(!(query_buffers_->getGroupByBuffersSize() == 0));
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_group_by_buffers =
        query_buffers_->createAndInitializeGroupByBufferGpu(ra_exe_unit,
                                                            query_mem_desc_,
                                                            executor_->getConfig(),
                                                            kernel_params[INIT_AGG_VALS],
                                                            device_id,
                                                            dispatch_mode_,
                                                            block_size_x,
                                                            grid_size_x,
                                                            executor_->warpSize(),
                                                            can_sort_on_gpu,
                                                            output_columnar_);
    const auto max_matched = static_cast<int32_t>(gpu_group_by_buffers.entry_count);
    buffer_provider->copyToDevice(reinterpret_cast<int8_t*>(kernel_params[MAX_MATCHED]),
                                  reinterpret_cast<const int8_t*>(&max_matched),
                                  sizeof(max_matched),
                                  device_id);

    kernel_params[GROUPBY_BUF] = gpu_group_by_buffers.ptrs;

    KernelOptions ko = {grid_size_x,
                        grid_size_y,
                        grid_size_z,
                        block_size_x,
                        block_size_y,
                        block_size_z,
                        static_cast<unsigned int>(shared_memory_size),
                        LITERALS,
                        hoist_literals};

    if (executor_->getConfig().exec.watchdog.enable_dynamic || allow_runtime_interrupt) {
      auto prepare_time = prepareClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by prepare: " << std::to_string(prepare_time)
              << " ms";
      launchClock->start();
    }
    auto timer_kernel = DEBUG_TIMER("Actual kernel");
    kernel->launch(ko, kernel_params);

    if (executor_->getConfig().exec.watchdog.enable_dynamic || allow_runtime_interrupt) {
      auto launch_time = launchClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by cuLaunchKernel: "
              << std::to_string(launch_time) << " ms";
      finishClock->start();
    }

    gpu_allocator_->copyFromDevice(reinterpret_cast<int8_t*>(error_codes.data()),
                                   reinterpret_cast<int8_t*>(err_desc),
                                   error_codes.size() * sizeof(error_codes[0]));
    timer_kernel.stop();
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }

    if (query_mem_desc_.useStreamingTopN()) {
      query_buffers_->applyStreamingTopNOffsetGpu(
          buffer_provider,
          query_mem_desc_,
          gpu_group_by_buffers,
          ra_exe_unit,
          total_thread_count,
          device_id,
          executor_->getConfig().exec.group_by.bigint_count);
    } else {
      if (use_speculative_top_n(ra_exe_unit, query_mem_desc_)) {
        try {
          inplace_sort_gpu(ra_exe_unit.sort_info.order_entries,
                           query_mem_desc_,
                           gpu_group_by_buffers,
                           buffer_provider,
                           device_id);
        } catch (const std::bad_alloc&) {
          throw SpeculativeTopNFailed("Failed during in-place GPU sort.");
        }
      }
      if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
        if (query_mem_desc_.didOutputColumnar()) {
          query_buffers_->compactProjectionBuffersGpu(
              query_mem_desc_,
              buffer_provider,
              gpu_group_by_buffers,
              get_num_allocated_rows_from_gpu(
                  buffer_provider, kernel_params[TOTAL_MATCHED], device_id),
              device_id);
        } else {
          query_buffers_->copyGroupByBuffersFromGpu(
              buffer_provider,
              query_mem_desc_,
              query_mem_desc_.getEntryCount(),
              gpu_group_by_buffers,
              &ra_exe_unit,
              block_size_x,
              grid_size_x,
              device_id,
              can_sort_on_gpu && query_mem_desc_.hasKeylessHash());
        }
      } else {
        query_buffers_->copyGroupByBuffersFromGpu(
            buffer_provider,
            query_mem_desc_,
            query_mem_desc_.getEntryCount(),
            gpu_group_by_buffers,
            &ra_exe_unit,
            block_size_x,
            grid_size_x,
            device_id,
            can_sort_on_gpu && query_mem_desc_.hasKeylessHash());
      }
    }
  } else {
    std::vector<int8_t*> out_vec_dev_buffers;
    const size_t agg_col_count{ra_exe_unit.estimator ? size_t(1) : init_agg_vals.size()};
    // by default, non-grouped aggregate queries generate one result per available thread
    // in the lifetime of (potentially multi-fragment) kernel execution.
    // We can reduce these intermediate results internally in the device and hence have
    // only one result per device, if GPU shared memory optimizations are enabled.
    const auto num_results_per_agg_col =
        shared_memory_size ? 1 : block_size_x * grid_size_x * num_fragments;
    const auto output_buffer_size_per_agg = num_results_per_agg_col * sizeof(int64_t);
    if (ra_exe_unit.estimator) {
      estimator_result_set_.reset(new ResultSet(
          ra_exe_unit.estimator, ExecutorDeviceType::GPU, device_id, data_mgr));
      out_vec_dev_buffers.push_back(
          reinterpret_cast<int8_t*>(estimator_result_set_->getDeviceEstimatorBuffer()));
    } else {
      for (size_t i = 0; i < agg_col_count; ++i) {
        int8_t* out_vec_dev_buffer =
            num_fragments ? reinterpret_cast<int8_t*>(
                                gpu_allocator_->alloc(output_buffer_size_per_agg))
                          : 0;
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
        if (shared_memory_size) {
          CHECK_EQ(output_buffer_size_per_agg, size_t(8));
          gpu_allocator_->copyToDevice(reinterpret_cast<int8_t*>(out_vec_dev_buffer),
                                       reinterpret_cast<const int8_t*>(&init_agg_vals[i]),
                                       output_buffer_size_per_agg);
        }
      }
    }
    auto out_vec_dev_ptr = gpu_allocator_->alloc(agg_col_count * sizeof(int8_t*));
    gpu_allocator_->copyToDevice(out_vec_dev_ptr,
                                 reinterpret_cast<int8_t*>(out_vec_dev_buffers.data()),
                                 agg_col_count * sizeof(int8_t*));
    kernel_params[GROUPBY_BUF] = reinterpret_cast<int8_t*>(out_vec_dev_ptr);
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    KernelOptions ko = {grid_size_x,
                        grid_size_y,
                        grid_size_z,
                        block_size_x,
                        block_size_y,
                        block_size_z,
                        static_cast<unsigned int>(shared_memory_size),
                        LITERALS,
                        hoist_literals};

    if (executor_->getConfig().exec.watchdog.enable_dynamic ||
        (allow_runtime_interrupt)) {
      auto prepare_time = prepareClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: prepare: " << std::to_string(prepare_time) << " ms";
      launchClock->start();
    }

    kernel->launch(ko, kernel_params);

    if (executor_->getConfig().exec.watchdog.enable_dynamic || allow_runtime_interrupt) {
      auto launch_time = launchClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: cuLaunchKernel: " << std::to_string(launch_time)
              << " ms";
      finishClock->start();
    }

    buffer_provider->copyFromDevice(reinterpret_cast<int8_t*>(&error_codes[0]),
                                    reinterpret_cast<const int8_t*>(err_desc),
                                    error_codes.size() * sizeof(error_codes[0]),
                                    device_id);
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }
    if (ra_exe_unit.estimator) {
      CHECK(estimator_result_set_);
      estimator_result_set_->syncEstimatorBuffer();
      return {};
    }
    for (size_t i = 0; i < agg_col_count; ++i) {
      int64_t* host_out_vec = new int64_t[output_buffer_size_per_agg];
      buffer_provider->copyFromDevice(
          reinterpret_cast<int8_t*>(host_out_vec),
          reinterpret_cast<const int8_t*>(out_vec_dev_buffers[i]),
          output_buffer_size_per_agg,
          device_id);
      out_vec.push_back(host_out_vec);
    }
  }
  const auto count_distinct_bitmap_mem = query_buffers_->getCountDistinctBitmapPtr();
  if (count_distinct_bitmap_mem) {
    buffer_provider->copyFromDevice(
        query_buffers_->getCountDistinctHostPtr(),
        reinterpret_cast<const int8_t*>(count_distinct_bitmap_mem),
        query_buffers_->getCountDistinctBitmapBytes(),
        device_id);
  }

  const auto varlen_output_gpu_buf = query_buffers_->getVarlenOutputPtr();
  if (varlen_output_gpu_buf) {
    CHECK(query_mem_desc_.varlenOutputBufferElemSize());
    const size_t varlen_output_buf_bytes =
        query_mem_desc_.getEntryCount() *
        query_mem_desc_.varlenOutputBufferElemSize().value();
    CHECK(query_buffers_->getVarlenOutputHostPtr());
    buffer_provider->copyFromDevice(
        query_buffers_->getVarlenOutputHostPtr(),
        reinterpret_cast<const int8_t*>(varlen_output_gpu_buf),
        varlen_output_buf_bytes,
        device_id);
  }

  if (executor_->getConfig().exec.watchdog.enable_dynamic || allow_runtime_interrupt) {
    auto finish_time = finishClock->stop();
    VLOG(1) << "Device " << std::to_string(device_id)
            << ": launchGpuCode: finish: " << std::to_string(finish_time) << " ms";
  }
  gpu_allocator_->free(
      reinterpret_cast<Data_Namespace::AbstractBuffer*>(kernel_metadata_gpu_buf));
  return out_vec;
}

std::vector<int64_t*> QueryExecutionContext::launchCpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CpuCompilationContext* native_code,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    int32_t* error_code,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables,
    const int64_t num_rows_to_process) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(lauchCpuCode);

  CHECK(query_buffers_);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  std::vector<const int8_t**> multifrag_col_buffers;
  for (auto& col_buffer : col_buffers) {
    multifrag_col_buffers.push_back(col_buffer.empty() ? nullptr : col_buffer.data());
  }
  const int8_t*** multifrag_cols_ptr{
      multifrag_col_buffers.empty() ? nullptr : &multifrag_col_buffers[0]};
  const uint64_t num_fragments =
      multifrag_cols_ptr ? static_cast<uint64_t>(col_buffers.size()) : uint64_t(0);
  const auto num_out_frags = multifrag_cols_ptr ? num_fragments : uint64_t(0);

  const bool is_group_by{query_mem_desc_.isGroupBy()};
  std::vector<int64_t*> out_vec;
  if (ra_exe_unit.estimator) {
    // Subfragments collect the result from multiple runs in a single
    // result set.
    if (!estimator_result_set_) {
      estimator_result_set_.reset(
          new ResultSet(ra_exe_unit.estimator, ExecutorDeviceType::CPU, 0, nullptr));
    }
    out_vec.push_back(
        reinterpret_cast<int64_t*>(estimator_result_set_->getHostEstimatorBuffer()));
  } else if (ra_exe_unit.isShuffle()) {
    out_vec.push_back(
        reinterpret_cast<int64_t*>(executor_->shuffle_out_buf_ptrs_.data()));
  } else {
    if (!is_group_by) {
      for (size_t i = 0; i < init_agg_vals.size(); ++i) {
        auto buff = new int64_t[num_out_frags];
        out_vec.push_back(static_cast<int64_t*>(buff));
      }
    }
  }

  CHECK_EQ(num_rows.size(), col_buffers.size());
  std::vector<int64_t> flatened_num_rows;
  for (auto& nums : num_rows) {
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  std::vector<uint64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    flatened_frag_offsets.insert(
        flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  int64_t rowid_lookup_num_rows{*error_code ? *error_code + 1 : 0};
  int64_t* num_rows_ptr;
  if (num_rows_to_process > 0) {
    flatened_num_rows[0] = num_rows_to_process;
    num_rows_ptr = flatened_num_rows.data();
  } else {
    num_rows_ptr =
        rowid_lookup_num_rows ? &rowid_lookup_num_rows : flatened_num_rows.data();
  }
  int32_t total_matched_init{0};

  std::vector<int64_t> cmpt_val_buff;
  if (is_group_by) {
    cmpt_val_buff =
        compact_init_vals(align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t),
                          init_agg_vals,
                          query_mem_desc_);
  }

  CHECK(native_code);
  const int64_t* join_hash_tables_ptr =
      join_hash_tables.size() == 1
          ? reinterpret_cast<int64_t*>(join_hash_tables[0])
          : (join_hash_tables.size() > 1 ? &join_hash_tables[0] : nullptr);
  int64_t** out_buffers = ra_exe_unit.isShuffle() || !is_group_by
                              ? out_vec.data()
                              : query_buffers_->getGroupByBuffersPtr();
  if (hoist_literals) {
    using agg_query = void (*)(const int8_t***,  // col_buffers
                               const uint64_t*,  // num_fragments
                               const int8_t*,    // literals
                               const int64_t*,   // num_rows
                               const uint64_t*,  // frag_row_offsets
                               const int32_t*,   // max_matched
                               int32_t*,         // total_matched
                               const int64_t*,   // init_agg_value
                               int64_t**,        // out
                               int32_t*,         // error_code
                               const uint32_t*,  // num_tables
                               const int64_t*);  // join_hash_tables_ptr
    if (is_group_by) {
      reinterpret_cast<agg_query>(native_code->func())(multifrag_cols_ptr,
                                                       &num_fragments,
                                                       literal_buff.data(),
                                                       num_rows_ptr,
                                                       flatened_frag_offsets.data(),
                                                       &scan_limit,
                                                       &total_matched_init,
                                                       cmpt_val_buff.data(),
                                                       out_buffers,
                                                       error_code,
                                                       &num_tables,
                                                       join_hash_tables_ptr);
    } else {
      reinterpret_cast<agg_query>(native_code->func())(multifrag_cols_ptr,
                                                       &num_fragments,
                                                       literal_buff.data(),
                                                       num_rows_ptr,
                                                       flatened_frag_offsets.data(),
                                                       &scan_limit,
                                                       &total_matched_init,
                                                       init_agg_vals.data(),
                                                       out_buffers,
                                                       error_code,
                                                       &num_tables,
                                                       join_hash_tables_ptr);
    }
  } else {
    using agg_query = void (*)(const int8_t***,  // col_buffers
                               const uint64_t*,  // num_fragments
                               const int64_t*,   // num_rows
                               const uint64_t*,  // frag_row_offsets
                               const int32_t*,   // max_matched
                               int32_t*,         // total_matched
                               const int64_t*,   // init_agg_value
                               int64_t**,        // out
                               int32_t*,         // error_code
                               const uint32_t*,  // num_tables
                               const int64_t*);  // join_hash_tables_ptr
    if (is_group_by) {
      reinterpret_cast<agg_query>(native_code->func())(multifrag_cols_ptr,
                                                       &num_fragments,
                                                       num_rows_ptr,
                                                       flatened_frag_offsets.data(),
                                                       &scan_limit,
                                                       &total_matched_init,
                                                       cmpt_val_buff.data(),
                                                       out_buffers,
                                                       error_code,
                                                       &num_tables,
                                                       join_hash_tables_ptr);
    } else {
      reinterpret_cast<agg_query>(native_code->func())(multifrag_cols_ptr,
                                                       &num_fragments,
                                                       num_rows_ptr,
                                                       flatened_frag_offsets.data(),
                                                       &scan_limit,
                                                       &total_matched_init,
                                                       init_agg_vals.data(),
                                                       out_buffers,
                                                       error_code,
                                                       &num_tables,
                                                       join_hash_tables_ptr);
    }
  }

  if (ra_exe_unit.estimator) {
    return {};
  }

  if (rowid_lookup_num_rows && *error_code < 0) {
    *error_code = 0;
  }

  if (query_mem_desc_.useStreamingTopN()) {
    query_buffers_->applyStreamingTopNOffsetCpu(query_mem_desc_, ra_exe_unit);
  }

  if (query_mem_desc_.didOutputColumnar() &&
      query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    query_buffers_->compactProjectionBuffersCpu(query_mem_desc_, total_matched_init);
  }
  return out_vec;
}

size_t QueryExecutionContext::getKernelParamsAllocSize(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int32_t>& error_codes,
    const size_t literals_sz,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const size_t init_agg_vals_sz,
    const size_t hash_table_count) const {
  // Scalar vals: NUM_FRAGMENTS + INIT_AGG_VALS + MAX_MATCHED + TOTAL_MATCHED + NUM_TABLES
  size_t total_alloc_size = sizeof(uint64_t) + sizeof(int64_t) + sizeof(int32_t) +
                            sizeof(int32_t) + sizeof(uint32_t);
  const uint64_t num_fragments = static_cast<uint64_t>(col_buffers.size());
  const size_t col_count{num_fragments > 0 ? col_buffers.front().size() : 0};
  if (col_count) {
    total_alloc_size += col_buffers.size() * col_count * sizeof(int8_t*) +
                        num_fragments * sizeof(int8_t*);
  }

  const size_t flatened_num_rows_sz = std::accumulate(
      num_rows.begin(),
      num_rows.end(),
      0,
      [](size_t sum, const std::vector<int64_t>& nums) { return sum + nums.size(); });

  const size_t flatened_frag_offsets_sz =
      std::accumulate(frag_offsets.begin(),
                      frag_offsets.end(),
                      0,
                      [](size_t sum, const std::vector<uint64_t>& offsets) {
                        return sum + offsets.size();
                      });

  total_alloc_size += sizeof(int64_t) * 2 + literals_sz;
  total_alloc_size += sizeof(int64_t) * flatened_num_rows_sz;
  total_alloc_size += sizeof(int64_t) * flatened_frag_offsets_sz;
  total_alloc_size += sizeof(int64_t) * init_agg_vals_sz;
  total_alloc_size += sizeof(error_codes[0]) * error_codes.size();
  if (hash_table_count > 1) {
    total_alloc_size += hash_table_count * sizeof(int64_t);
  }
  if (total_alloc_size % 8) {
    total_alloc_size += (8 - (total_alloc_size % 8));
  }
  LOG(DEBUG1) << total_alloc_size << " bytes on GPU for kernel params";
  return total_alloc_size;
}

// Allocates one buffer for parameters and fills in decreasing alignment of parameter
// types.
std::pair<std::vector<int8_t*>, Data_Namespace::AbstractBuffer*>
QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables,
    BufferProvider* buffer_provider,
    const int device_id,
    const bool hoist_literals,
    const bool is_group_by) const {
  INJECT_TIMER(prepareKernelParams);
  CHECK(gpu_allocator_);
  std::vector<int8_t*> params(KERN_PARAM_COUNT, 0);

  CHECK_EQ(num_rows.size(), col_buffers.size());
  CHECK_EQ(frag_offsets.size(), col_buffers.size());

  std::vector<int64_t> additional_literal_bytes;
  const auto count_distinct_bitmap_mem = query_buffers_->getCountDistinctBitmapPtr();
  if (count_distinct_bitmap_mem) {
    // Store host and device addresses
    const auto count_distinct_bitmap_host_mem = query_buffers_->getCountDistinctHostPtr();
    CHECK(count_distinct_bitmap_host_mem);
    additional_literal_bytes.push_back(
        reinterpret_cast<int64_t>(count_distinct_bitmap_host_mem));
    additional_literal_bytes.push_back(static_cast<int64_t>(
        reinterpret_cast<std::uintptr_t>(count_distinct_bitmap_mem)));
  }

  const int8_t* init_agg_vals_buf{nullptr};
  size_t init_agg_vals_buf_sz{0};
  std::vector<int64_t> compact_init_vals_vec;
  if (is_group_by && !output_columnar_) {
    init_agg_vals_buf_sz =
        align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    compact_init_vals_vec =
        compact_init_vals(init_agg_vals_buf_sz, init_agg_vals, query_mem_desc_);
    init_agg_vals_buf = reinterpret_cast<const int8_t*>(compact_init_vals_vec.data());
  } else {
    init_agg_vals_buf = reinterpret_cast<const int8_t*>(init_agg_vals.data());
    init_agg_vals_buf_sz = init_agg_vals.size();
  }

  const auto hash_table_count = join_hash_tables.size();

  const size_t alloc_size{
      getKernelParamsAllocSize(col_buffers,
                               error_codes,
                               literal_buff.size() + additional_literal_bytes.size(),
                               num_rows,
                               frag_offsets,
                               init_agg_vals_buf_sz,
                               hash_table_count)};

  Data_Namespace::AbstractBuffer* kernel_metadata_gpu_buf =
      gpu_allocator_->allocGpuAbstractBuffer(alloc_size);
  int8_t* kernel_metadata_gpu_cursor{kernel_metadata_gpu_buf->getMemoryPtr()};
  auto copy_to_gpu_mem = [device_id, buffer_provider, &kernel_metadata_gpu_cursor](
                             const int8_t* from, const size_t size) {
    buffer_provider->copyToDeviceAsyncIfPossible(
        kernel_metadata_gpu_cursor, from, size, device_id);
    int8_t* copied_to = kernel_metadata_gpu_cursor;
    kernel_metadata_gpu_cursor += size;
    return copied_to;
  };

  const uint64_t num_fragments = static_cast<uint64_t>(col_buffers.size());
  const size_t col_count{num_fragments > 0 ? col_buffers.front().size() : 0};
  std::vector<int8_t*> multifrag_col_dev_buffers;
  std::vector<const int8_t*> flatened_col_buffers;
  if (col_count) {
    std::vector<size_t> col_buffs_offsets;
    for (auto& buffers : col_buffers) {
      flatened_col_buffers.insert(
          flatened_col_buffers.end(), buffers.begin(), buffers.end());
      col_buffs_offsets.push_back(
          col_buffs_offsets.size() ? col_buffs_offsets.back() + buffers.size() : 0);
    }
    int8_t* col_buffers_dev_ptr =
        copy_to_gpu_mem(reinterpret_cast<const int8_t*>(flatened_col_buffers.data()),
                        flatened_col_buffers.size() * sizeof(int8_t*));
    for (const size_t offset : col_buffs_offsets) {
      multifrag_col_dev_buffers.push_back(
          reinterpret_cast<int8_t*>(col_buffers_dev_ptr + offset * sizeof(int8_t*)));
    }
    params[COL_BUFFERS] =
        copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&multifrag_col_dev_buffers[0]),
                        num_fragments * sizeof(int8_t*));
  }

  params[NUM_FRAGMENTS] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&num_fragments), sizeof(uint64_t));
  params[INIT_AGG_VALS] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(init_agg_vals_buf),
                      init_agg_vals_buf_sz * sizeof(int64_t));
  std::vector<int64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    CHECK_EQ(offsets.size(), num_tables);
    flatened_frag_offsets.insert(
        flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  params[FRAG_ROW_OFFSETS] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&flatened_frag_offsets[0]),
                      sizeof(int64_t) * flatened_frag_offsets.size());

  std::vector<int64_t> flatened_num_rows;
  for (auto& nums : num_rows) {
    CHECK_EQ(nums.size(), num_tables);
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  params[NUM_ROWS] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(flatened_num_rows.data()),
                      sizeof(int64_t) * flatened_num_rows.size());

  switch (hash_table_count) {
    case 0: {
      params[JOIN_HASH_TABLES] = 0;
      break;
    }
    case 1:
      params[JOIN_HASH_TABLES] = reinterpret_cast<int8_t*>(join_hash_tables[0]);
      break;
    default: {
      params[JOIN_HASH_TABLES] =
          copy_to_gpu_mem(reinterpret_cast<const int8_t*>(join_hash_tables.data()),
                          hash_table_count * sizeof(int64_t));
      break;
    }
  }

  int8_t* literals_and_addr_mapping = kernel_metadata_gpu_cursor;
  CHECK_EQ(std::uintptr_t(0),
           reinterpret_cast<std::uintptr_t>(literals_and_addr_mapping) % 8);
  if (count_distinct_bitmap_mem) {
    copy_to_gpu_mem(
        reinterpret_cast<const int8_t*>(&additional_literal_bytes[0]),
        additional_literal_bytes.size() * sizeof(additional_literal_bytes[0]));
  }
  params[LITERALS] = literals_and_addr_mapping + additional_literal_bytes.size() *
                                                     sizeof(additional_literal_bytes[0]);
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&literal_buff[0]),
                    literal_buff.size());
  }

  if (reinterpret_cast<std::uintptr_t>(kernel_metadata_gpu_cursor) % 4) {
    kernel_metadata_gpu_cursor +=
        (4 - (reinterpret_cast<std::uintptr_t>(kernel_metadata_gpu_cursor) % 4));
  }
  CHECK_EQ(std::uintptr_t(0),
           reinterpret_cast<std::uintptr_t>(kernel_metadata_gpu_cursor) % 4);
  // Note that this will be overwritten if we are setting the entry count during group by
  // buffer allocation and initialization
  const int32_t max_matched{scan_limit};
  int32_t total_matched{0};

  params[ERROR_CODE] = copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&error_codes[0]),
                                       error_codes.size() * sizeof(error_codes[0]));
  params[MAX_MATCHED] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&max_matched), sizeof(max_matched));
  params[TOTAL_MATCHED] = copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&total_matched),
                                          sizeof(total_matched));
  params[NUM_TABLES] =
      copy_to_gpu_mem(reinterpret_cast<const int8_t*>(&num_tables), sizeof(uint32_t));
  CHECK_LE(reinterpret_cast<std::uintptr_t>(kernel_metadata_gpu_cursor),
           reinterpret_cast<std::uintptr_t>(kernel_metadata_gpu_buf->getMemoryPtr() +
                                            alloc_size));
  CHECK_EQ(nullptr, params[GROUPBY_BUF]);
  buffer_provider->synchronizeDeviceDataStream(device_id);

  return {params, kernel_metadata_gpu_buf};
}
