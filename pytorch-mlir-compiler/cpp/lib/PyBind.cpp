#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ai_compiler/Dialect.h"
#include "ai_compiler/Passes.h"
#include "ai_compiler/Pipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace py = pybind11;

using SerializedIR = std::map<std::string, py::object>;

static std::string compileFromSerializedIr(const SerializedIR &ir, const std::string &target) {
  (void)ir;
  (void)target;

  mlir::MLIRContext context;
  context.loadDialect<mlir::ai::AIDialect>();
  mlir::registerLLVMDialectTranslation(context);

  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::PassManager pm(&context);
  mlir::ai::registerAIPasses();
  mlir::ai::buildAIPipeline(pm);

  if (mlir::failed(pm.run(module))) {
    throw std::runtime_error("Failed to run AI pipeline");
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    throw std::runtime_error("Failed to translate to LLVM IR");
  }

  std::string llvmIR;
  llvm::raw_string_ostream os(llvmIR);
  llvmModule->print(os, nullptr);
  os.flush();
  return llvmIR;
}

PYBIND11_MODULE(_ai_compiler_mlir, m) {
  m.def("compile_from_serialized_ir", &compileFromSerializedIr, py::arg("ir"), py::arg("target") = "cpu");
}

