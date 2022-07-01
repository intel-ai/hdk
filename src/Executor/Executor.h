#pragma once

#include <vector>

#include "Codegen/Context.h"
#include "Kernel/Kernel.h"

namespace hdk {
class Device;
class SomeKindOfBuffer;
// a lightweight executor, device specific?
// the entity, fullfiling the requirement that kernels are executed identically
// connects to storage? via buffers?
// ie. per kernel/kernel sequence;
class Executor {
 public:
  using KernelFunction = Context;  // kind of..
  using MemoryBuffers = std::vector<SomeKindOfBuffer>;
  using InputData = SomeKindOfBuffer;
  using OutputData = SomeKindOfBuffer;

  Executor(Kernel&, Device);  // or different executor type?
  // run method should probably just create a task and submit it to a scheduler ????
  // gets the code from kernel function, accesses device memory manager and runs the
  // workload passing appropriate arguments.
  OutputData run(InputData&, KernelFunction&, MemoryBuffers&);
};

}  // namespace hdk
