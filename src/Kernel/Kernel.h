#pragma once

namespace hdk {
class FragmentDesc;
struct Kernel {
  FragmentDesc id;
  // anything else?
};

struct KernelSequence;  // a DAG of Kernels
}  // namespace hdk
