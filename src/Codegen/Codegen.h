#pragma once

#include <memory>
#include "Backend.h"
#include "Context.h"

namespace hdk {
class ExpressionIR;  // or smth, step

// device independent, gets device details from a backend
// consider smem code generation, is backend enough/too ugly?
// Compiler? ExpressionCompiler?
class CodeGenerator {
 public:
  CodeGenerator(const Backend& backend);
  // goes through IR hierarchy if present, runs optimizations and then calls
  // backend.generateNativeCode() rename?
  std::shared_ptr<Context> generateNativeCode(const ExpressionIR&) const;
};
}  // namespace hdk
