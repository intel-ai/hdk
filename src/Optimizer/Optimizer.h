#pragma once

namespace hdk {
class StepSequence;
// prob better to use a llvm-style pass manager.
class Optimizer {
 public:
  StepSequence& optimize(StepSequence&);
};

}  // namespace hdk
