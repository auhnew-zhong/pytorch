#include "ai_compiler/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::ai;

namespace {

struct AISimplifyPass : public PassWrapper<AISimplifyPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "ai-simplify"; }
  StringRef getDescription() const final { return "Simplify ai dialect operations"; }
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<Pass> mlir::ai::createAISimplifyPass() { return std::make_unique<AISimplifyPass>(); }

void mlir::ai::registerAIPasses() {
  static PassRegistration<AISimplifyPass> reg;
  (void)reg;
}
