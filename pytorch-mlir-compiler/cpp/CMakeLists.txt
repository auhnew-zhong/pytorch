add_library(ai_compiler
  lib/Dialect.cpp
  lib/Passes.cpp
  lib/Pipeline.cpp
  lib/PyBind.cpp
)

target_include_directories(ai_compiler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(ai_compiler
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRLLVMToLLVMIRTranslation
    MLIRSupport
    Python3::Python
)

mlir_check_all_link_libraries(ai_compiler)

