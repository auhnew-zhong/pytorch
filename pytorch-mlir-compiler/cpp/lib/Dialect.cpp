#include "ai_compiler/Dialect.h"

using namespace mlir;
using namespace mlir::ai;

AIDialect::AIDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<AIDialect>()) {}

