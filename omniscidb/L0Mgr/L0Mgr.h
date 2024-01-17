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

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "DataMgr/GpuMgr.h"
#include "L0Mgr/L0Exception.h"
#include "L0Mgr/Utils.h"
#include "Logger/Logger.h"

#ifdef HAVE_L0
#include <level_zero/ze_api.h>
#endif

namespace l0 {

class L0Device;
class L0Driver {
 private:
  std::vector<std::shared_ptr<L0Device>> devices_;

#ifdef HAVE_L0
  ze_context_handle_t context_;
  ze_driver_handle_t driver_;
#endif

 public:
#ifdef HAVE_L0
  explicit L0Driver(ze_driver_handle_t handle);
  ze_context_handle_t ctx() const;
  ze_driver_handle_t driver() const;
  ~L0Driver();
#endif

  const std::vector<std::shared_ptr<L0Device>>& devices() const;
};

class L0Module;
class L0Kernel;
class L0CommandList;
class L0CommandQueue;

class L0Device {
 protected:
  const L0Driver& driver_;
  std::shared_ptr<L0CommandQueue> command_queue_;
#ifdef HAVE_L0
  ze_device_handle_t device_;
  ze_device_properties_t props_;
  ze_device_compute_properties_t compute_props_;
#endif
 private:
  /*
    This component for data fetching to L0 devices is used for
    more efficient asynchronous data transfers. It allows to amortize
    the costs of data transfers which is especially useful in case of
    many relatively small data transfers.
  */
  class L0DataFetcher {
#ifdef HAVE_L0
    static constexpr size_t GRAVEYARD_LIMIT{500};
    static constexpr size_t CL_BYTES_LIMIT{128 * 1024 * 1024};
    static constexpr ze_command_list_desc_t cl_desc_ = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        nullptr,
        0,
        ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT};
    struct CLBytesTracker {
      ze_command_list_handle_t cl_handle_;
      uint64_t bytes_;
    };
    CLBytesTracker cur_cl_bytes_;
    ze_command_queue_handle_t queue_handle_;
    std::list<ze_command_list_handle_t> graveyard_;
    std::list<ze_command_list_handle_t> recycled_;
    std::mutex cur_cl_lock_;
#endif
    L0Device& my_device_;
    void recycleGraveyard();
    void setCLRecycledOrNew();

   public:
    void appendCopyCommand(void* dst, const void* src, const size_t num_bytes);
    void sync();

#ifdef HAVE_L0
    L0DataFetcher(L0Device& device);
    ~L0DataFetcher();
#else
    L0DataFetcher() = default;
#endif
  };
  L0DataFetcher data_fetcher_;

 public:
  std::shared_ptr<L0CommandQueue> command_queue() const;
  std::unique_ptr<L0CommandList> create_command_list() const;

  void transferToDevice(void* dst, const void* src, const size_t num_bytes);
  void syncDataTransfers();

  std::shared_ptr<L0Module> create_module(uint8_t* code,
                                          size_t len,
                                          bool log = false) const;

#ifdef HAVE_L0
  L0Device(const L0Driver& driver, ze_device_handle_t device);
  uint32_t maxGroupCount() const;
  uint32_t maxGroupSize() const;
  uint32_t maxSharedLocalMemory() const;
  ze_device_handle_t device() const;
  ze_context_handle_t ctx() const;
  ~L0Device();
#else
  L0Device() = default;
#endif
};

class L0Module : public std::enable_shared_from_this<L0Module> {
 public:
  std::shared_ptr<L0Kernel> create_kernel(const char* name,
                                          uint32_t x,
                                          uint32_t y,
                                          uint32_t z) const;
#ifdef HAVE_L0
  static std::shared_ptr<L0Module> make(ze_module_handle_t handle) {
    return std::shared_ptr<L0Module>(new L0Module(handle));
  }
  ze_module_handle_t handle() const;
  ~L0Module();

 private:
  ze_module_handle_t handle_;

  L0Module(ze_module_handle_t handle) : handle_(handle){};
#endif
};

#ifdef HAVE_L0
template <int Index>
void set_kernel_args(ze_kernel_handle_t kernel) {}

template <int Index, typename Head>
void set_kernel_args(ze_kernel_handle_t kernel, Head&& head) {
  L0_SAFE_CALL(zeKernelSetArgumentValue(
      kernel, Index, sizeof(std::remove_reference_t<Head>), head));
}

template <int Index, typename Head, typename... Tail>
void set_kernel_args(ze_kernel_handle_t kernel, Head&& head, Tail&&... tail) {
  set_kernel_args<Index>(kernel, head);
  set_kernel_args<Index + 1>(kernel, std::forward<Tail>(tail)...);
}
#endif

class L0Kernel {
 public:
#ifdef HAVE_L0
  L0Kernel(std::shared_ptr<const L0Module> parent,
           ze_kernel_handle_t handle,
           uint32_t x,
           uint32_t y,
           uint32_t z);
  ze_kernel_handle_t handle() const;
  std::string desc() const;
  ~L0Kernel();

 private:
  std::shared_ptr<const L0Module> parent_;
  ze_kernel_handle_t handle_;
#endif
};

class L0CommandQueue {
#ifdef HAVE_L0
 private:
  ze_command_queue_handle_t handle_;

 public:
  L0CommandQueue(ze_command_queue_handle_t handle);
  ze_command_queue_handle_t handle() const;
  ~L0CommandQueue();
#endif
};

struct GroupCount {
  uint32_t groupCountX;
  uint32_t groupCountY;
  uint32_t groupCountZ;
};

class L0CommandList {
 private:
#ifdef HAVE_L0
  ze_command_list_handle_t handle_;
#endif

 public:
  void copy(void* dst, const void* src, const size_t num_bytes);
  template <typename... Args>
  void launch(L0Kernel& kernel, const GroupCount& gc, Args&&... args) {
#ifdef HAVE_L0
    set_kernel_args<0>(kernel.handle(), std::forward<Args>(args)...);

    ze_group_count_t group_count = {gc.groupCountX, gc.groupCountY, gc.groupCountZ};
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
        handle_, kernel.handle(), &group_count, nullptr, 0, nullptr));

    L0_SAFE_CALL(zeCommandListAppendBarrier(handle_, nullptr, 0, nullptr));
#endif
  }

  void launch(L0Kernel* kernel, std::vector<int8_t*>& params, const GroupCount& gc);

  void submit(L0CommandQueue& queue);

#ifdef HAVE_L0
  ze_command_list_handle_t handle() const;
  L0CommandList(ze_command_list_handle_t handle);
  ~L0CommandList();
#endif
};

void* allocate_device_mem(const size_t num_bytes, L0Device& device);

class L0Manager : public GpuMgr {
 public:
  L0Manager();
  void copyHostToDeviceAsync(int8_t* device_ptr,
                             const int8_t* host_ptr,
                             const size_t num_bytes,
                             const int device_num) override;

  void copyHostToDeviceAsyncIfPossible(int8_t* device_ptr,
                                       const int8_t* host_ptr,
                                       const size_t num_bytes,
                                       const int device_num) override;

  void synchronizeDeviceDataStream(const int device_num) override;

  void copyHostToDevice(int8_t* device_ptr,
                        const int8_t* host_ptr,
                        const size_t num_bytes,
                        const int device_num) override;
  void copyDeviceToHost(int8_t* host_ptr,
                        const int8_t* device_ptr,
                        const size_t num_bytes,
                        const int device_num) override;
  void copyDeviceToDevice(int8_t* dest_ptr,
                          int8_t* src_ptr,
                          const size_t num_bytes,
                          const int dest_device_num,
                          const int src_device_num) override;

  int8_t* allocatePinnedHostMem(const size_t num_bytes);
  int8_t* allocateDeviceMem(const size_t num_bytes, const int device_num) override;
  void freePinnedHostMem(int8_t* host_ptr);
  void freeDeviceMem(int8_t* device_ptr) override;
  void zeroDeviceMem(int8_t* device_ptr,
                     const size_t num_bytes,
                     const int device_num) override;
  void setDeviceMem(int8_t* device_ptr,
                    const unsigned char uc,
                    const size_t num_bytes,
                    const int device_num) override;
  bool canLoadAsync() const override { return async_data_load_available; };
  void synchronizeDevices() const override;
  GpuMgrPlatform getPlatform() const override { return GpuMgrPlatform::L0; }
  int getDeviceCount() const override {
    return drivers_.size() ? drivers_[0]->devices().size() : 0;
  }
  void setContext(const int device_num) const override{
      // nothing to do here as the actual context is explicitly passed as a parameter
      // to every manager's method
  };

  size_t getMaxAllocationSize(const int device_num) const;
  size_t getTotalMem(const int device_num) const override;
  size_t getPageSize(const int device_num) const { return 4096u; }

  uint32_t getMaxBlockSize() const override;
  int8_t getSubGroupSize() const override;
  uint32_t getGridSize() const override;
  uint32_t getMinEUNumForAllDevices() const override;
  bool hasSharedMemoryAtomicsSupport() const override;
  bool hasFP64Support() const override;
  size_t getMinSharedMemoryPerBlockForAllDevices() const override;

  const std::vector<std::shared_ptr<L0Driver>>& drivers() const;

 private:
  std::vector<std::shared_ptr<L0Driver>> drivers_;
  static constexpr bool async_data_load_available{true};
};

}  // namespace l0
