#pragma once

#include <memory>
#include <vector>

#include "Codegen/Context.h"
#include "Optimizer/Optimizer.h"

namespace hdk {
class ExpressionIR;

class Engine {
  std::vector<std::shared_ptr<Context>> contexts;

 public:
  void run_query(ExpressionIR& ir) {
    auto step_seq = generate_step_seq(
        ir);  // or do we expect ExpressionIR to be exactly that sequence?
    auto opt = Optimizer();
    // ... add opt passes
    auto& result = opt.optimize(
        step_seq);  // resulting graph should contain kernels and device tags for those.

    auto scheduler = ...;
    scheduler.algorithm = ...;

    // prepare an execution DAG: compile here?
    for (auto& step : result) {
      step.prepare(contexts);
    }

    // create tasks for kernels and schedule
    // should we make the executor - a task?
    // - create tasks and submit to executor?
    // - create executors and submit to scheduler?
    // - ???
    for (auto& step : result) {
      scheduler.queue.add(Executor(step.kernel, step.device));
    }
  }
};

}  // namespace hdk
