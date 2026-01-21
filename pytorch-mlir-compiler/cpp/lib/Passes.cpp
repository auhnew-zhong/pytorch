#include "ai_compiler/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::ai;

namespace {

struct AISimplifyPass : public PassWrapper<AISimplifyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<Pass> mlir::ai::createAISimplifyPass() { return std::make_unique<AISimplifyPass>(); }

void mlir::ai::registerAIPasses() {
  PassRegistration<AISimplifyPass>("ai-simplify", "Simplify ai dialect operations");
}

