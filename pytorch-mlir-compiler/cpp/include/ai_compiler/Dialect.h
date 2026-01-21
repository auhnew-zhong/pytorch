#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace ai {

class AIDialect : public Dialect {
public:
  explicit AIDialect(MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "ai"; }
};

} // namespace ai
} // namespace mlir

