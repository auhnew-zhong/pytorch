#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace ai {

void buildAIPipeline(PassManager &pm);

} // namespace ai
} // namespace mlir

