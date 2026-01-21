#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ai {

std::unique_ptr<Pass> createAISimplifyPass();
void registerAIPasses();

} // namespace ai
} // namespace mlir

