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

#pragma once

#include "Logger/Logger.h"
#include "QueryEngine/LLVMGlobalContext.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/IR/Module.h>

struct CompilationOptions;

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

inline std::string llvmErrorToString(const llvm::Error& err) {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << err;
  return msg;
};

class ORCJITExecutionEngineWrapper {
 public:
  ORCJITExecutionEngineWrapper(
      std::unique_ptr<llvm::orc::ExecutionSession>&& execution_session,
      llvm::orc::JITTargetMachineBuilder target_machine_builder,
      std::unique_ptr<llvm::DataLayout> data_layout)
      : execution_session_(std::move(execution_session))
      , data_layout_(std::move(data_layout))
      , mangle_(std::make_unique<llvm::orc::MangleAndInterner>(*this->execution_session_,
                                                               *data_layout_))
      , object_layer_(std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
            *execution_session_,
            []() { return std::make_unique<llvm::SectionMemoryManager>(); }))
      , compiler_layer_(std::make_unique<llvm::orc::IRCompileLayer>(
            *execution_session_,
            *object_layer_,
            std::make_unique<llvm::orc::ConcurrentIRCompiler>(
                std::move(target_machine_builder)))) {
#ifdef _WIN32
    object_layer_->setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_->setAutoClaimResponsibilityForObjectSymbols(true);
#endif
    auto dylib_or_error = execution_session_->createJITDylib("<main>");
    if (!dylib_or_error) {
      LOG(FATAL) << "Failed to initialize JITTargetMachineBuilder: "
                 << llvmErrorToString(dylib_or_error.takeError());
    }
    main_dylib_ = &(*dylib_or_error);
    dylib_or_error->addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_->getGlobalPrefix())));
  }

  ~ORCJITExecutionEngineWrapper() { llvm::cantFail(execution_session_->endSession()); }

  ORCJITExecutionEngineWrapper(const ORCJITExecutionEngineWrapper& other) = delete;
  ORCJITExecutionEngineWrapper(ORCJITExecutionEngineWrapper&& other) = delete;

  void addModule(std::unique_ptr<llvm::Module> module) {
    module->setDataLayout(*data_layout_);
    llvm::orc::ThreadSafeModule tsm(std::move(module), getGlobalLLVMThreadSafeContext());
    auto err = compiler_layer_->add(*main_dylib_, std::move(tsm));
    if (err) {
      LOG(FATAL) << "Cannot add LLVM module: " << llvmErrorToString(err);
    }
  }

  void* getPointerToFunction(llvm::Function* function) {
    CHECK(function);
    CHECK(execution_session_);
    auto symbol =
        execution_session_->lookup({main_dylib_}, (*mangle_)(function->getName()));
    if (!symbol) {
      LOG(FATAL) << "Failed to find function " << std::string(function->getName())
                 << "\nError: " << llvmErrorToString(symbol.takeError());
    }
    return reinterpret_cast<void*>(symbol->getAddress());
  }

  bool exists() const { return !(execution_session_ == nullptr); }

  void removeModule(llvm::Module* module) {
    // Do nothing here. Module is deleted by ORC after materialization.
  }

  ORCJITExecutionEngineWrapper& operator=(const ORCJITExecutionEngineWrapper& other) =
      delete;
  ORCJITExecutionEngineWrapper& operator=(ORCJITExecutionEngineWrapper&& other) = delete;

 private:
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  std::unique_ptr<llvm::DataLayout> data_layout_;
  std::unique_ptr<llvm::orc::MangleAndInterner> mangle_;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  std::unique_ptr<llvm::orc::IRCompileLayer> compiler_layer_;
  std::unique_ptr<llvm::JITEventListener> intel_jit_listener_;

  llvm::orc::JITDylib* main_dylib_;
};

using ExecutionEngineWrapper = ORCJITExecutionEngineWrapper;
