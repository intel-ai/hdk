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

#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef __CUDACC__
#define HOST __host__
#else
#define HOST
#endif

#ifdef __CUDACC__
#define GLOBAL __global__
#else
#define GLOBAL
#endif

#if defined(__CUDACC__) && __CUDACC_VER_MAJOR__ < 8
#define STATIC_QUAL
#else
#define STATIC_QUAL static
#endif

#ifdef __CUDACC__
#define FORCE_INLINE __forceinline__
#elif defined(_WIN32)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#if defined(__CUDACC__) || (defined(__GNUC__) && defined(__SANITIZE_THREAD__)) || \
    defined(WITH_JIT_DEBUG)
#define ALWAYS_INLINE
#elif defined(_WIN32)
#define ALWAYS_INLINE
#elif defined(ENABLE_SHARED_LIBS)
// Protected visibility allows to inline non-private functions which can
// be overwritten at dynamic link time with the default visibility.
#define ALWAYS_INLINE \
  __attribute__((always_inline)) __attribute__((__visibility__("protected")))
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#ifdef L0_RUNTIME_ENABLED
#define GENERIC_ADDR_SPACE __attribute__((opencl_generic))
#define GLOBAL_ADDR_SPACE __attribute__((opencl_global))
#define SHARED_ADDR_SPACE __attribute__((opencl_local))
#define NOREMOVE __attribute__((optnone))
#else
#define GENERIC_ADDR_SPACE
#define GLOBAL_ADDR_SPACE
#define SHARED_ADDR_SPACE
#define NOREMOVE
#endif

#ifdef __CUDACC__
#define NEVER_INLINE
#elif defined(_WIN32)
#define NEVER_INLINE __declspec(noinline)
#else
#define NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#define SUFFIX(name) name##_gpu
#else
#define SUFFIX(name) name
#endif

#ifdef _WIN32
#define RUNTIME_EXPORT __declspec(dllexport)
#define EXTERN __declspec(dllimport)
#else
#define RUNTIME_EXPORT
#define EXTERN
#endif
