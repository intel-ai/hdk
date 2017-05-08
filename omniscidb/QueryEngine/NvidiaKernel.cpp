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

#include "NvidiaKernel.h"

#include "../Shared/mapdpath.h"

#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

#ifdef HAVE_CUDA
namespace {

void fill_options(std::vector<CUjit_option>& option_keys,
                  std::vector<void*>& option_values,
                  const unsigned block_size_x) {
  option_keys.push_back(CU_JIT_LOG_VERBOSE);
  option_values.push_back(reinterpret_cast<void*>(1));
  option_keys.push_back(CU_JIT_THREADS_PER_BLOCK);
  option_values.push_back(reinterpret_cast<void*>(block_size_x));
}

}  // namespace

CubinResult ptx_to_cubin(const std::string& ptx,
                         const unsigned block_size,
                         const CudaMgr_Namespace::CudaMgr* cuda_mgr) {
  CHECK(!ptx.empty());
  CHECK(cuda_mgr->getDeviceCount() > 0);
  static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr)->setContext(0);
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  fill_options(option_keys, option_values, block_size);
  CHECK_EQ(option_values.size(), option_keys.size());
  unsigned num_options = option_keys.size();
  CUlinkState link_state;
  checkCudaErrors(cuLinkCreate(num_options, &option_keys[0], &option_values[0], &link_state));

  boost::filesystem::path gpu_rt_path{mapd_root_abs_path()};
  gpu_rt_path /= "QueryEngine";
  gpu_rt_path /= "cuda_mapd_rt.a";
  if (!boost::filesystem::exists(gpu_rt_path)) {
    throw std::runtime_error("MapD GPU runtime library not found at " + gpu_rt_path.string());
  }

  if (!gpu_rt_path.empty()) {
    // How to create a static CUDA library:
    // 1. nvcc -std=c++11 -arch=sm_30 --device-link -c [list of .cu files]
    // 2. nvcc -std=c++11 -arch=sm_30 -lib [list of .o files generated by step 1] -o [library_name.a]
    checkCudaErrors(cuLinkAddFile(
        link_state, CU_JIT_INPUT_LIBRARY, gpu_rt_path.c_str(), num_options, &option_keys[0], &option_values[0]));
  }
  checkCudaErrors(cuLinkAddData(link_state,
                                CU_JIT_INPUT_PTX,
                                static_cast<void*>(const_cast<char*>(ptx.c_str())),
                                ptx.length() + 1,
                                0,
                                num_options,
                                &option_keys[0],
                                &option_values[0]));
  void* cubin{nullptr};
  size_t cubinSize{0};
  checkCudaErrors(cuLinkComplete(link_state, &cubin, &cubinSize));
  CHECK(cubin);
  CHECK_GT(cubinSize, size_t(0));
  return {cubin, option_keys, option_values, link_state};
}
#endif

#ifdef HAVE_CUDA
GpuCompilationContext::GpuCompilationContext(const void* image,
                                             const std::string& kernel_name,
                                             const int device_id,
                                             const void* cuda_mgr,
                                             unsigned int num_options,
                                             CUjit_option* options,
                                             void** option_vals)
    : module_(nullptr), kernel_(nullptr), device_id_(device_id), cuda_mgr_(cuda_mgr) {
  static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr_)->setContext(device_id_);
  checkCudaErrors(cuModuleLoadDataEx(&module_, image, num_options, options, option_vals));
  CHECK(module_);
  checkCudaErrors(cuModuleGetFunction(&kernel_, module_, kernel_name.c_str()));
}
#endif  // HAVE_CUDA

GpuCompilationContext::~GpuCompilationContext() {
#ifdef HAVE_CUDA
  static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr_)->setContext(device_id_);
  auto status = cuModuleUnload(module_);
  // TODO(alex): handle this race better
  if (status == CUDA_ERROR_DEINITIALIZED) {
    return;
  }
  checkCudaErrors(status);
#endif
}
