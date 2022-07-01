#include "HDK.h"

int main() {
  /*
   * interaction steps:
   * 1. Lower the input workflow into Expression IR sequence - get type information (user
   * side, not hdk functionality)
   *   - we only allow user to build the IR via Expression IR iface. Validation?
   * 2. Create optimizer and populate with optimizations, optimize Expression IR sequence
   *   - optimizer should know what devices are we targeting to make a choice of data
   *     distribution
   *   - optimizer output should either be of the same type as Expression IR sequence or
   *     the lowered step DAG
   *   - specific optimization passes:
   *       - merge pass unites steps that can be done withing a single loop (no pipeline
   *         breakers)
   *       - device scheduling pass - for each step decides device distribution, and
   *         embeds this metadata into step
   *       - some passes require additional metainfo, like cardinality estimation - need a
   *         similar to llvm mechanism
   * 3. Transform the step sequence into execution graph (we called this phase a
   * "scheduler", but it's not)
   *   - create kernels and assign them to devices according to the distribution using the
   *     optimizer hint.
   *   - at this stage the graph should be ready to run (e.g. with velox)
   *   - allow user to create their own kernels?
   *   - allow user to manipulate the distribution (how to preserve consistency?)
   * 4. Submit the graph to the executor
   *   - the executor should know what scheduler (or policy) user wants it to use
   *   - the exeuctor creates tasks (?) and submits them to the scheduler
   *   - user gets a notification on completion/can query execution status
   */

  // 1
  auto builder = hdk::IR::ExpressionIRBuilder();
  auto step = builder.addStep();  // similar to a BB?
  step.makeColumnExpr(...);
  //....
  auto ir_seq = builder.commit();
  assert(hdk::IR::validate(ir_seq));

  // 2
  auto optimizer = hdk::OptimizerManager(hdk::device::USE_ALL);
  optimizer.add(hdk::opt::MergePass());
  optimizer.add(hdk::opt::DeviceSchedulingPass());
  auto steps = optimizer.run(ir_seq);  // probably shouldn't return anything

  // 3
  auto exec_dag = hdk::create_kernels(steps, inputs);  // do we need another entity here?
                                                       // graph of pairs {kernel, device}
                                                       // should be self-sufficient?
                                                       // this one needs further analysis

  // 4. option 1 - built-in JIT
  auto executor = hdk::Executor();
  auto future = executor.submit(exec_dag);
  auto result = future.wait();

  // 4. option 2 - explicit compilation

  // this replaces JIT-Engine entity and stores compilation contexts
  // how does executor retrive it?
  auto contexts = hdk::ContextStorage();

  // in case user wants their own compiler
  for (auto {kernel, device} : exec_dag) {
    auto backend = hdk::make_backend(device);
    auto compiler = hdk::make_compiler(backend);
    // or thier own compiler that returns compilation context
    contexts.push(compiler.compile(kernel));
  }

  auto executor = hdk::Executor(contexts);
  auto future = executor.submit(exec_dag);
  auto result = future.wait();

  /* Summary */
  //   Such api allows :
  //   * building arbitraty expression sequences
  //   * adding custom optimizations at expression IR level
  //   * apply custom lowering and compilers from expression IR
  //   * submit execution graph to any available executor?
}