/*
 * Copyright 2020 OmniSci, Inc.
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

#include "L0Mgr/L0Mgr.h"

#include "Logger/Logger.h"
#include "Utils.h"

#include <algorithm>
#include <iostream>
#include <limits>

#include <level_zero/ze_api.h>

namespace l0 {

L0Driver::L0Driver(ze_driver_handle_t handle) : driver_(handle) {
  ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  L0_SAFE_CALL(zeContextCreate(driver_, &ctx_desc, &context_));

  uint32_t device_count = 0;
  L0_SAFE_CALL(zeDeviceGet(driver_, &device_count, nullptr));

  std::vector<ze_device_handle_t> devices(device_count);
  L0_SAFE_CALL(zeDeviceGet(driver_, &device_count, devices.data()));

  for (auto device : devices) {
    ze_device_properties_t device_properties;
    L0_SAFE_CALL(zeDeviceGetProperties(device, &device_properties));
    if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
      devices_.push_back(std::make_shared<L0Device>(*this, device));
    }
  }
}

L0Driver::~L0Driver() {
  auto status = (zeContextDestroy(context_));
  if (status) {
    LOG(ERROR) << "Non-zero status for context destructor";
  }
}

ze_context_handle_t L0Driver::ctx() const {
  return context_;
}

ze_driver_handle_t L0Driver::driver() const {
  return driver_;
}

const std::vector<std::shared_ptr<L0Device>>& L0Driver::devices() const {
  return devices_;
}

std::vector<std::shared_ptr<L0Driver>> get_drivers() {
  L0_SAFE_CALL(zeInit(0));
  uint32_t driver_count = 0;
  L0_SAFE_CALL(zeDriverGet(&driver_count, nullptr));

  std::vector<ze_driver_handle_t> handles(driver_count);
  L0_SAFE_CALL(zeDriverGet(&driver_count, handles.data()));

  LOG(INFO) << "Discovered " << driver_count << " driver(s) for L0 platform.";

  std::vector<std::shared_ptr<L0Driver>> result(driver_count);
  for (uint32_t i = 0; i < driver_count; i++) {
    result[i] = std::make_shared<L0Driver>(handles[i]);
  }
  return result;
}

L0CommandList::L0CommandList(ze_command_list_handle_t handle) : handle_(handle) {}

void L0CommandList::launch(L0Kernel* kernel,
                           std::vector<int8_t*>& params,
                           const GroupCount& gc) {
  for (unsigned i = 0; i < params.size(); ++i) {
    L0_SAFE_CALL(zeKernelSetArgumentValue(
        kernel->handle(), i, sizeof(params[i]), params[i] ? &params[i] : nullptr));
  }

  LOG(INFO) << "L0 kernel group count: {" << gc.groupCountX << "," << gc.groupCountY
            << "," << gc.groupCountZ << "}\n";
  ze_group_count_t group_count = {gc.groupCountX, gc.groupCountY, gc.groupCountZ};
  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
      handle_, kernel->handle(), &group_count, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(handle_, nullptr, 0, nullptr));
}

void L0CommandList::submit(L0CommandQueue& queue) {
  L0_SAFE_CALL(zeCommandListClose(handle_));
  L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(queue.handle(), 1, &handle_, nullptr));
  L0_SAFE_CALL(
      zeCommandQueueSynchronize(queue.handle(), std::numeric_limits<uint32_t>::max()));
}

ze_command_list_handle_t L0CommandList::handle() const {
  return handle_;
}

L0CommandList::~L0CommandList() {
  // TODO: maybe return to pool
  auto status = zeCommandListDestroy(handle_);
  if (status) {
    LOG(ERROR) << "Non-zero status for command list destructor";
  }
}

void L0CommandList::copy(void* dst, const void* src, const size_t num_bytes) {
  L0_SAFE_CALL(
      zeCommandListAppendMemoryCopy(handle_, dst, src, num_bytes, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(handle_, nullptr, 0, nullptr));
}

void* allocate_device_mem(const size_t num_bytes, L0Device& device) {
  ze_device_mem_alloc_desc_t alloc_desc;
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  alloc_desc.pNext = nullptr;
  alloc_desc.flags = 0;
  alloc_desc.ordinal = 0;

  void* mem;
  L0_SAFE_CALL(zeMemAllocDevice(
      device.ctx(), &alloc_desc, num_bytes, 0 /*align*/, device.device(), &mem));
  return mem;
}

L0Device::L0DataFetcher::L0DataFetcher(L0Device& device) : my_device_(device) {
  ze_command_queue_desc_t command_queue_fetch_desc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      0,
      0,
      0,
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  L0_SAFE_CALL(zeCommandQueueCreate(my_device_.driver_.ctx(),
                                    my_device_.device_,
                                    &command_queue_fetch_desc,
                                    &queue_handle_));
  cur_cl_bytes_ = {{}, 0};
  L0_SAFE_CALL(zeCommandListCreate(my_device_.driver_.ctx(),
                                   my_device_.device_,
                                   &cl_desc_,
                                   &cur_cl_bytes_.cl_handle_));
}

L0Device::L0DataFetcher::~L0DataFetcher() {
  zeCommandQueueDestroy(queue_handle_);
  zeCommandListDestroy(cur_cl_bytes_.cl_handle_);
  for (auto& dead_handle : graveyard_) {
    zeCommandListDestroy(dead_handle);
  }
  for (auto& cl_handle : recycled_) {
    zeCommandListDestroy(cl_handle);
  }
}

void L0Device::L0DataFetcher::recycleGraveyard() {
  while (recycled_.size() < GRAVEYARD_LIMIT && graveyard_.size()) {
    recycled_.push_back(graveyard_.front());
    graveyard_.pop_front();
    L0_SAFE_CALL(zeCommandListReset(recycled_.back()));
  }
  for (auto& dead_handle : graveyard_) {
    L0_SAFE_CALL(zeCommandListDestroy(dead_handle));
  }
  graveyard_.clear();
}

void L0Device::L0DataFetcher::setCLRecycledOrNew() {
  cur_cl_bytes_ = {{}, 0};
  if (recycled_.size()) {
    cur_cl_bytes_.cl_handle_ = recycled_.front();
    recycled_.pop_front();
  } else {
    L0_SAFE_CALL(zeCommandListCreate(my_device_.driver_.ctx(),
                                     my_device_.device_,
                                     &cl_desc_,
                                     &cur_cl_bytes_.cl_handle_));
  }
}

void L0Device::L0DataFetcher::appendCopyCommand(void* dst,
                                                const void* src,
                                                const size_t num_bytes) {
  std::unique_lock<std::mutex> cl_lock(cur_cl_lock_);
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      cur_cl_bytes_.cl_handle_, dst, src, num_bytes, nullptr, 0, nullptr));
  cur_cl_bytes_.bytes_ += num_bytes;
  if (cur_cl_bytes_.bytes_ >= CL_BYTES_LIMIT) {
    ze_command_list_handle_t cl_h_copy = cur_cl_bytes_.cl_handle_;
    graveyard_.push_back(cur_cl_bytes_.cl_handle_);
    setCLRecycledOrNew();
    cl_lock.unlock();
    L0_SAFE_CALL(zeCommandListClose(cl_h_copy));
    L0_SAFE_CALL(
        zeCommandQueueExecuteCommandLists(queue_handle_, 1, &cl_h_copy, nullptr));
  }
}

void L0Device::L0DataFetcher::sync() {
  if (cur_cl_bytes_.bytes_) {
    L0_SAFE_CALL(zeCommandListClose(cur_cl_bytes_.cl_handle_));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(
        queue_handle_, 1, &cur_cl_bytes_.cl_handle_, nullptr));
  }
  L0_SAFE_CALL(
      zeCommandQueueSynchronize(queue_handle_, std::numeric_limits<uint32_t>::max()));
  L0_SAFE_CALL(zeCommandListReset(cur_cl_bytes_.cl_handle_));
  if (graveyard_.size() > GRAVEYARD_LIMIT) {
    recycleGraveyard();
  }
}

L0Device::L0Device(const L0Driver& driver, ze_device_handle_t device)
    : device_(device), driver_(driver), data_fetcher_(*this) {
  ze_command_queue_handle_t queue_handle;
  ze_command_queue_desc_t command_queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                                nullptr,
                                                0,
                                                0,
                                                0,
                                                ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  L0_SAFE_CALL(
      zeCommandQueueCreate(driver_.ctx(), device_, &command_queue_desc, &queue_handle));
  L0_SAFE_CALL(zeDeviceGetProperties(device_, &props_));
  CHECK_EQ(ZE_DEVICE_TYPE_GPU, props_.type);
  L0_SAFE_CALL(zeDeviceGetComputeProperties(device_, &compute_props_));

  command_queue_ = std::make_shared<L0CommandQueue>(queue_handle);
}

L0Device::~L0Device() {}

ze_context_handle_t L0Device::ctx() const {
  return driver_.ctx();
}
ze_device_handle_t L0Device::device() const {
  return device_;
}
std::shared_ptr<L0CommandQueue> L0Device::command_queue() const {
  return command_queue_;
}

std::unique_ptr<L0CommandList> L0Device::create_command_list() const {
  ze_command_list_desc_t desc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
      nullptr,
      0,
      0  // flags
  };
  ze_command_list_handle_t res;
  L0_SAFE_CALL(zeCommandListCreate(ctx(), device_, &desc, &res));
  return std::make_unique<L0CommandList>(res);
}

uint32_t L0Device::maxGroupCount() const {
  return props_.numSlices * props_.numSubslicesPerSlice;
}
uint32_t L0Device::maxGroupSize() const {
  return compute_props_.maxGroupSizeX;
}
unsigned L0Device::maxSharedLocalMemory() const {
  return compute_props_.maxSharedLocalMemory;
}

void L0Device::transferToDevice(void* dst, const void* src, const size_t num_bytes) {
  data_fetcher_.appendCopyCommand(dst, src, num_bytes);
}

void L0Device::syncDataTransfers() {
  data_fetcher_.sync();
}

L0CommandQueue::L0CommandQueue(ze_command_queue_handle_t handle) : handle_(handle) {}

ze_command_queue_handle_t L0CommandQueue::handle() const {
  return handle_;
}

L0CommandQueue::~L0CommandQueue() {
  auto status = (zeCommandQueueDestroy(handle_));
  if (status) {
    LOG(ERROR) << "Non-zero status for command queue destructor";
  }
}

std::shared_ptr<L0Module> L0Device::create_module(uint8_t* code,
                                                  size_t len,
                                                  bool log) const {
  ze_module_desc_t desc{
      .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
      .pNext = nullptr,
      .format = ZE_MODULE_FORMAT_IL_SPIRV,
      .inputSize = len,
      .pInputModule = code,
      .pBuildFlags = "",
      .pConstants = nullptr,
  };
  ze_module_handle_t handle;
  ze_module_build_log_handle_t buildlog = nullptr;
  auto status = zeModuleCreate(ctx(), device_, &desc, &handle, &buildlog);
  if (log) {
    size_t logSize = 0;
    L0_SAFE_CALL(zeModuleBuildLogGetString(buildlog, &logSize, nullptr));
    std::string strLog(logSize, ' ');
    L0_SAFE_CALL(zeModuleBuildLogGetString(buildlog, &logSize, strLog.data()));
    LOG(INFO) << "L0 module build log: " << strLog;
  }
  L0_SAFE_CALL(zeModuleBuildLogDestroy(buildlog));
  if (status) {
    throw l0::L0Exception(status);
  }
  return L0Module::make(handle);
}

L0Manager::L0Manager() : drivers_(get_drivers()) {}

const std::vector<std::shared_ptr<L0Driver>>& L0Manager::drivers() const {
  return drivers_;
}

int8_t* L0Manager::allocateDeviceMem(const size_t num_bytes, int device_id) {
  auto& device = drivers_[0]->devices()[device_id];
  return (int8_t*)allocate_device_mem(num_bytes, *device);
}

ze_module_handle_t L0Module::handle() const {
  return handle_;
}

L0Module::~L0Module() {
  auto status = zeModuleDestroy(handle_);
  if (status) {
    LOG(ERROR) << "Non-zero status for command module destructor";
  }
}

std::shared_ptr<L0Kernel> L0Module::create_kernel(const char* name,
                                                  uint32_t x,
                                                  uint32_t y,
                                                  uint32_t z) const {
  ze_kernel_desc_t desc{
      .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
      .pNext = nullptr,
      .flags = 0,
      .pKernelName = name,
  };
  ze_kernel_handle_t handle;
  L0_SAFE_CALL(zeKernelCreate(this->handle_, &desc, &handle));
  return std::make_shared<L0Kernel>(shared_from_this(), handle, x, y, z);
}

L0Kernel::L0Kernel(std::shared_ptr<const L0Module> parent,
                   ze_kernel_handle_t handle,
                   uint32_t x,
                   uint32_t y,
                   uint32_t z)
    : parent_(parent), handle_(handle) {
  LOG(INFO) << "Setting group size: {" << x << "," << y << "," << z << "}\n";
  L0_SAFE_CALL(zeKernelSetGroupSize(handle_, x, y, z));
}

ze_kernel_handle_t L0Kernel::handle() const {
  return handle_;
}

std::string L0Kernel::desc() const {
  ze_kernel_properties_t props;
  L0_SAFE_CALL(zeKernelGetProperties(handle_, &props));
  std::ostringstream os;
  os << "kernel:{numargs:" << props.numKernelArgs
     << ",reqGroupSizeX:" << props.requiredGroupSizeX
     << ",reqGroupSizeY:" << props.requiredGroupSizeY
     << ",reqGroupSizeZ:" << props.requiredGroupSizeZ
     << ",maxNumSubgroups:" << props.maxNumSubgroups
     << ",maxSubgroupSize:" << props.maxSubgroupSize
     << ",privateMemSize:" << props.privateMemSize
     << ",requiredNumSubGroups:" << props.requiredNumSubGroups
     << ",requiredSubgroupSize:" << props.requiredSubgroupSize << "}";
  return os.str();
}

L0Kernel::~L0Kernel() {
  auto status = zeKernelDestroy(handle_);
  if (status) {
    LOG(ERROR) << "Non-zero status for command kernel destructor";
  }
}

void L0Manager::copyHostToDevice(int8_t* device_ptr,
                                 const int8_t* host_ptr,
                                 const size_t num_bytes,
                                 const int device_num) {
  if (!num_bytes)
    return;
  CHECK(host_ptr);
  CHECK(device_ptr);
  CHECK_GT(num_bytes, 0);
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());

  auto& device = drivers()[0]->devices()[device_num];
  auto cl = device->create_command_list();
  auto queue = device->command_queue();

  cl->copy(device_ptr, host_ptr, num_bytes);
  cl->submit(*queue);
}

void L0Manager::copyHostToDeviceAsync(int8_t* device_ptr,
                                      const int8_t* host_ptr,
                                      const size_t num_bytes,
                                      const int device_num) {
  if (!num_bytes)
    return;
  CHECK(host_ptr);
  CHECK(device_ptr);
  CHECK_GT(num_bytes, 0);
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());

  auto& device = drivers()[0]->devices()[device_num];
  device->transferToDevice(device_ptr, host_ptr, num_bytes);
}

void L0Manager::copyHostToDeviceAsyncIfPossible(int8_t* device_ptr,
                                                const int8_t* host_ptr,
                                                const size_t num_bytes,
                                                const int device_num) {
  if constexpr (async_data_load_available) {
    copyHostToDeviceAsync(device_ptr, host_ptr, num_bytes, device_num);
  } else {
    copyHostToDevice(device_ptr, host_ptr, num_bytes, device_num);
  }
}

void L0Manager::synchronizeDeviceDataStream(const int device_num) {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());
  auto& device = drivers()[0]->devices()[device_num];
  device->syncDataTransfers();
}

void L0Manager::copyDeviceToHost(int8_t* host_ptr,
                                 const int8_t* device_ptr,
                                 const size_t num_bytes,
                                 const int device_num) {
  if (!num_bytes)
    return;
  CHECK(host_ptr);
  CHECK(device_ptr);
  CHECK_GT(num_bytes, 0);
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());

  auto& device = drivers_[0]->devices()[device_num];
  auto cl = device->create_command_list();
  auto queue = device->command_queue();

  cl->copy(host_ptr, device_ptr, num_bytes);
  cl->submit(*queue);
}

void L0Manager::copyDeviceToDevice(int8_t* dest_ptr,
                                   int8_t* src_ptr,
                                   const size_t num_bytes,
                                   const int dest_device_num,
                                   const int src_device_num) {
  CHECK(false);
}

int8_t* L0Manager::allocatePinnedHostMem(const size_t num_bytes) {
  CHECK(false);
  return nullptr;
}

void L0Manager::freePinnedHostMem(int8_t* host_ptr) {
  CHECK(false);
}

void L0Manager::freeDeviceMem(int8_t* device_ptr) {
  CHECK(device_ptr);
  auto ctx = drivers_[0]->ctx();
  L0_SAFE_CALL(zeMemFree(ctx, device_ptr));
}

void L0Manager::zeroDeviceMem(int8_t* device_ptr,
                              const size_t num_bytes,
                              const int device_num) {
  if (!num_bytes)
    return;
  CHECK(device_ptr);
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());
  CHECK_GE(num_bytes, 0);
  setDeviceMem(device_ptr, 0, num_bytes, device_num);
}
void L0Manager::setDeviceMem(int8_t* device_ptr,
                             const unsigned char uc,
                             const size_t num_bytes,
                             const int device_num) {
  if (!num_bytes)
    return;
  CHECK(device_ptr);
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());
  CHECK_GE(num_bytes, 0);
  auto& device = drivers_[0]->devices()[device_num];
  auto cl = device->create_command_list();
  L0_SAFE_CALL(zeCommandListAppendMemoryFill(
      cl->handle(), device_ptr, &uc, 1, num_bytes, nullptr, 0, nullptr));
  cl->submit(*device->command_queue());
}

void L0Manager::synchronizeDevices() const {
  for (auto& device : drivers_[0]->devices()) {
    L0_SAFE_CALL(zeCommandQueueSynchronize(device->command_queue()->handle(),
                                           std::numeric_limits<uint32_t>::max()));
  }
}

size_t L0Manager::getMaxAllocationSize(const int device_num) const {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());
  ze_device_properties_t device_properties;
  L0_SAFE_CALL(zeDeviceGetProperties(drivers_[0]->devices()[device_num]->device(),
                                     &device_properties));
  CHECK_EQ(ZE_DEVICE_TYPE_GPU, device_properties.type);
  LOG(INFO) << "Intel GPU max memory allocation size: "
            << device_properties.maxMemAllocSize / (1024 * 1024) << "MB\n";
  return device_properties.maxMemAllocSize;
}

size_t L0Manager::getTotalMem(const int device_num) const {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, drivers_[0]->devices().size());
  uint32_t p_count{0};
  L0_SAFE_CALL(zeDeviceGetMemoryProperties(
      drivers_[0]->devices()[device_num]->device(), &p_count, nullptr));
  CHECK_GT(p_count, 0u);
  std::vector<ze_device_memory_properties_t> mem_properties;
  mem_properties.resize(p_count);
  L0_SAFE_CALL(zeDeviceGetMemoryProperties(
      drivers_[0]->devices()[device_num]->device(), &p_count, mem_properties.data()));
  return static_cast<size_t>(mem_properties[0].totalSize);
}

uint32_t L0Manager::getMaxBlockSize() const {
  CHECK_GT(drivers_[0]->devices().size(), size_t(0));
  unsigned sz = drivers_[0]->devices()[0]->maxGroupSize();
  for (auto d : drivers_[0]->devices()) {
    sz = d->maxGroupSize() < sz ? d->maxGroupSize() : sz;
  }
  CHECK_GT(sz, 0);
  return sz;
}

int8_t L0Manager::getSubGroupSize() const {
  return 1;
}

uint32_t L0Manager::getGridSize() const {
  CHECK_GT(drivers_[0]->devices().size(), size_t(0));
  auto cnt = drivers_[0]->devices()[0]->maxGroupCount();
  for (auto d : drivers_[0]->devices()) {
    cnt = d->maxGroupCount() < cnt ? d->maxGroupCount() : cnt;
  }
  CHECK_GT(cnt, 0);
  return cnt;
}

uint32_t L0Manager::getMinEUNumForAllDevices() const {
  return 1u;
}

bool L0Manager::hasSharedMemoryAtomicsSupport() const {
  return true;
}

bool L0Manager::hasFP64Support() const {
  CHECK_GT(drivers_[0]->devices().size(), size_t(0));
  ze_device_module_properties_t module_props{ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES};
  L0_SAFE_CALL(
      zeDeviceGetModuleProperties(drivers_[0]->devices()[0]->device(), &module_props));
  return module_props.fp64flags;
}

size_t L0Manager::getMinSharedMemoryPerBlockForAllDevices() const {
  auto comp = [](const auto& a, const auto& b) {
    return a->maxSharedLocalMemory() < b->maxSharedLocalMemory();
  };
  return std::min_element(
             drivers_[0]->devices().begin(), drivers_[0]->devices().end(), comp)
      ->get()
      ->maxSharedLocalMemory();
};

}  // namespace l0
