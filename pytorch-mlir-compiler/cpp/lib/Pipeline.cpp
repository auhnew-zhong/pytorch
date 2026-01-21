#include "ai_compiler/Pipeline.h"

#include "ai_compiler/Passes.h"

using namespace mlir;
using namespace mlir::ai;

void mlir::ai::buildAIPipeline(PassManager &pm) { pm.addPass(createAISimplifyPass()); }

