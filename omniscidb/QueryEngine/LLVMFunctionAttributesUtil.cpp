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

#include "LLVMFunctionAttributesUtil.h"

void mark_function_always_inline(llvm::Function* func, llvm::LLVMContext& context) {
  func->addAttributeAtIndex(llvm::AttributeList::AttrIndex::FunctionIndex,
                            llvm::Attribute::get(context, llvm::Attribute::AlwaysInline));
}

void mark_function_never_inline(llvm::Function* func, llvm::LLVMContext& context) {
  clear_function_attributes(func);
  func->addAttributeAtIndex(llvm::AttributeList::AttrIndex::FunctionIndex,
                            llvm::Attribute::get(context, llvm::Attribute::NoInline));
}

void clear_function_attributes(llvm::Function* func) {
  llvm::AttributeList no_attributes;
  func->setAttributes(no_attributes);
}
