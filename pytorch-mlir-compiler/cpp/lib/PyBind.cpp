#define PYBIND11_NO_RTTI
#include <memory>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ai_compiler/Dialect.h"
#include "ai_compiler/Passes.h"
#include "ai_compiler/Pipeline.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace py = pybind11;

static std::string sanitizeSymbolName(std::string name) {
  for (char &c : name) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    if (!ok)
      c = '_';
  }
  if (name.empty() || ((name[0] < 'a' || name[0] > 'z') && (name[0] < 'A' || name[0] > 'Z') && name[0] != '_')) {
    name.insert(name.begin(), '_');
  }
  return name;
}

static mlir::Type parseElementType(const std::string &dtype, mlir::OpBuilder &builder) {
  if (dtype == "torch.float16")
    return builder.getF16Type();
  if (dtype == "torch.float32")
    return builder.getF32Type();
  if (dtype == "torch.float64")
    return builder.getF64Type();
  if (dtype == "torch.int32")
    return builder.getI32Type();
  if (dtype == "torch.int64")
    return builder.getI64Type();
  if (dtype == "torch.bool")
    return builder.getI1Type();
  return builder.getF32Type();
}

static mlir::Type parseValueType(const py::dict &value, mlir::OpBuilder &builder) {
  const std::string dtype = value.contains("dtype") ? py::str(value["dtype"]) : "torch.float32";
  mlir::Type elemType = parseElementType(dtype, builder);
  if (!value.contains("shape") || value["shape"].is_none()) {
    return mlir::UnrankedTensorType::get(elemType);
  }

  py::list shapeList = value["shape"].cast<py::list>();
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(shapeList.size());
  for (py::handle d : shapeList) {
    dims.push_back(d.cast<int64_t>());
  }
  return mlir::RankedTensorType::get(dims, elemType);
}

static std::string compileFromSerializedIr(py::dict ir, const std::string &target) {
  (void)target;

  mlir::MLIRContext context;
  context.loadDialect<mlir::ai::AIDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  if (!ir.contains("inputs") || !ir.contains("nodes") || !ir.contains("outputs")) {
    throw std::runtime_error("Serialized IR must contain inputs/nodes/outputs");
  }

  py::list inputsList = ir["inputs"].cast<py::list>();
  py::list nodesList = ir["nodes"].cast<py::list>();
  py::list outputsList = ir["outputs"].cast<py::list>();

  std::unordered_map<std::string, mlir::Type> valueTypes;
  std::vector<std::string> inputNames;
  llvm::SmallVector<mlir::Type, 4> inputTypes;

  inputNames.reserve(inputsList.size());
  inputTypes.reserve(inputsList.size());
  for (py::handle h : inputsList) {
    py::dict v = py::reinterpret_borrow<py::dict>(h);
    const std::string name = py::str(v["name"]);
    mlir::Type ty = parseValueType(v, builder);
    inputNames.push_back(name);
    inputTypes.push_back(ty);
    valueTypes[name] = ty;
  }

  std::vector<std::string> outputNames;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  outputNames.reserve(outputsList.size());
  outputTypes.reserve(outputsList.size());
  for (py::handle h : outputsList) {
    py::dict v = py::reinterpret_borrow<py::dict>(h);
    const std::string name = py::str(v["name"]);
    mlir::Type ty = parseValueType(v, builder);
    outputNames.push_back(name);
    outputTypes.push_back(ty);
    valueTypes[name] = ty;
  }

  mlir::Location loc = builder.getUnknownLoc();
  mlir::func::FuncOp mainFunc = mlir::func::FuncOp::create(loc, "main", builder.getFunctionType(inputTypes, outputTypes));
  module.push_back(mainFunc);

  mlir::Block *entry = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  std::unordered_map<std::string, mlir::Value> values;
  values.reserve(inputNames.size() + nodesList.size());
  for (size_t i = 0; i < inputNames.size(); ++i) {
    values[inputNames[i]] = entry->getArgument(i);
  }

  std::unordered_map<std::string, mlir::func::FuncOp> calleeCache;

  for (py::handle h : nodesList) {
    py::dict n = py::reinterpret_borrow<py::dict>(h);
    const std::string op = py::str(n["op"]);
    py::list ins = n["inputs"].cast<py::list>();
    py::list outs = n["outputs"].cast<py::list>();

    llvm::SmallVector<mlir::Value, 4> operands;
    operands.reserve(ins.size());
    llvm::SmallVector<mlir::Type, 4> operandTypes;
    operandTypes.reserve(ins.size());
    for (py::handle inNameHandle : ins) {
      const std::string inName = py::str(inNameHandle);
      auto it = values.find(inName);
      if (it == values.end()) {
        throw std::runtime_error(("Unknown input value name: " + inName).c_str());
      }
      operands.push_back(it->second);
      operandTypes.push_back(it->second.getType());
    }

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    resultTypes.reserve(outs.size());
    for (py::handle outNameHandle : outs) {
      const std::string outName = py::str(outNameHandle);
      auto itTy = valueTypes.find(outName);
      if (itTy != valueTypes.end()) {
        resultTypes.push_back(itTy->second);
      } else if (!operandTypes.empty()) {
        resultTypes.push_back(operandTypes.front());
        valueTypes[outName] = operandTypes.front();
      } else {
        mlir::Type fallback = mlir::UnrankedTensorType::get(builder.getF32Type());
        resultTypes.push_back(fallback);
        valueTypes[outName] = fallback;
      }
    }

    const std::string calleeKey = op + "|" + std::to_string(operandTypes.size()) + "->" + std::to_string(resultTypes.size());
    mlir::func::FuncOp callee;
    auto itCallee = calleeCache.find(calleeKey);
    if (itCallee != calleeCache.end()) {
      callee = itCallee->second;
    } else {
      const std::string calleeName = sanitizeSymbolName(op) + "__" + std::to_string(operandTypes.size()) + "_" + std::to_string(resultTypes.size());
      auto existing = module.lookupSymbol<mlir::func::FuncOp>(calleeName);
      if (existing) {
        callee = existing;
      } else {
        callee = mlir::func::FuncOp::create(loc, calleeName, builder.getFunctionType(operandTypes, resultTypes));
        callee.setPrivate();
        module.push_back(callee);
      }
      calleeCache.emplace(calleeKey, callee);
    }

    auto call = builder.create<mlir::func::CallOp>(loc, callee, operands);
    for (size_t i = 0; i < outs.size(); ++i) {
      const std::string outName = py::str(outs[i]);
      values[outName] = call.getResult(i);
    }
  }

  llvm::SmallVector<mlir::Value, 4> rets;
  rets.reserve(outputNames.size());
  for (const std::string &name : outputNames) {
    auto it = values.find(name);
    if (it == values.end()) {
      throw std::runtime_error(("Unknown output value name: " + name).c_str());
    }
    rets.push_back(it->second);
  }
  builder.create<mlir::func::ReturnOp>(loc, rets);

  mlir::PassManager pm(&context);
  mlir::ai::registerAIPasses();
  mlir::ai::buildAIPipeline(pm);

  if (mlir::failed(pm.run(module))) {
    throw std::runtime_error("Failed to run AI pipeline");
  }

  std::string mlirText;
  llvm::raw_string_ostream os(mlirText);
  module.print(os);
  os.flush();
  return mlirText;
}

PYBIND11_MODULE(_ai_compiler_mlir, m) {
  m.def("compile_from_serialized_ir", &compileFromSerializedIr, py::arg("ir"), py::arg("target") = "cpu");
}
