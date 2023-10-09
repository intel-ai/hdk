/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <gtest/gtest.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <LLVMSPIRVLib/LLVMSPIRVLib.h>

TEST(SPIRVBuildTest, TranslateSimple) {
  using namespace llvm;
  // See source at https://github.com/kurapov-peter/L0Snippets
  LLVMContext ctx;
  std::unique_ptr<Module> module = std::make_unique<Module>("code_generated", ctx);
  module->setTargetTriple("spir-unknown-unknown");
  IRBuilder<> builder(ctx);

  std::vector<Type*> args{PointerType::get(ctx, 1),
                          PointerType::get(ctx, 1)};  // float*, float*
  FunctionType* f_type = FunctionType::get(Type::getVoidTy(ctx), args, false);
  Function* f = Function::Create(
      f_type, GlobalValue::LinkageTypes::ExternalLinkage, "plus1", module.get());
  f->setCallingConv(CallingConv::SPIR_KERNEL);

  // get_global_id
  FunctionType* ggi_type =
      FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);
  Function* get_global_idj = Function::Create(ggi_type,
                                              GlobalValue::LinkageTypes::ExternalLinkage,
                                              "_Z13get_global_idj",
                                              module.get());
  get_global_idj->setCallingConv(CallingConv::SPIR_FUNC);

  BasicBlock* entry = BasicBlock::Create(ctx, "entry", f);

  builder.SetInsertPoint(entry);
  Constant* zero = ConstantInt::get(Type::getInt32Ty(ctx), 0);
  Constant* onef = ConstantFP::get(ctx, APFloat(1.f));
  Value* idx = builder.CreateCall(get_global_idj, zero, "idx");
  auto argit = f->args().begin();
  Value* firstElemSrc = builder.CreateGEP(Type::getFloatTy(ctx), argit, idx, "src.idx");
  ++argit;
  Value* firstElemDst = builder.CreateGEP(Type::getFloatTy(ctx), argit, idx, "dst.idx");
  Value* ldSrc = builder.CreateLoad(Type::getFloatTy(ctx), firstElemSrc, "ld");
  Value* result = builder.CreateFAdd(ldSrc, onef, "foo");
  builder.CreateStore(result, firstElemDst);
  builder.CreateRetVoid();

  // set metadata -- pretend we're opencl (see
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst#spir-v-instructions-mapped-to-llvm-metadata)
  Metadata* spirv_src_ops[] = {
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 3 /*OpenCL_C*/)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(ctx), 102000 /*OpenCL ver 1.2*/))};
  NamedMDNode* spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(MDNode::get(ctx, spirv_src_ops));

  module->print(errs(), nullptr);

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::ostringstream ss;
  std::string err;
  auto success = writeSpirv(module.get(), opts, ss, err);
  ASSERT_EQ(success, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
