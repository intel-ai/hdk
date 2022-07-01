#pragma once

namespace hdk {
// device specific, provides, say, pointer types, function signature attributes, etc.
// would probably like to hide llvm here?
class Backend {
 public:
  // using DevicePointerTy = ...;
  virtual DevicePointerTy getPointerTy() const = 0;
  virtual generateNativeCode() const = 0;
};

class CPUBackend : public Backend {};
class GPUBackend : public Backend {};
class XPUBackend : public Backend {};
// ....

/*i.e. compilation flow
auto IR = make_some_IR();

auto backend = hdk::make_cpu_backend();
auto codegen = hdk::CodeGenerator(backend);

auto context = codegen.codegen(IR);

//the context is then stored in "Jit-engine"
*/

}  // namespace hdk