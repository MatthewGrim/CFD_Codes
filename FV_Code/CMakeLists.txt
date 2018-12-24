cmake_minimum_required(VERSION 3.6.3)
project(VOF)

set(CMAKE_CXX_STANDARD 11)

include_directories(src/ /usr/local/Cellar/vtk/8.1.2/include/vtk-8.1 /usr/local/include)
link_directories(/usr/local/lib /usr/local/Cellar/vtk/8.1.2/lib)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

# file(GLOB_RECURSE SOURCE_FILES "src" "*.h" "*.cpp")

set(SOURCE_FILES
        src/testing/vof/algo/riemann_solvers/ExactRiemannSolver_unittest.cpp
        src/testing/vof/data/geom/Point_unittest.cpp
        src/testing/vof/data/geom/Vector_unittest.cpp
        src/testing/vof/data/grid/EulerianGrid_unittest.cpp
        src/vof/algo/boundaryConditions/IBoundaryCondition.h
        src/vof/algo/boundaryConditions/TransmissiveBoundaryCondition.cpp
        src/vof/algo/boundaryConditions/TransmissiveBoundaryCondition.h
        src/vof/data/eos/eosModel/IdealGasEquationOfState.cpp
        src/vof/data/eos/eosModel/IdealGasEquationOfState.h
        src/vof/data/eos/eosModel/IEquationOfState.h
        src/vof/algo/fluxCalculators/FluxCalculatorGodunovImpl.cpp
        src/vof/algo/fluxCalculators/FluxCalculatorGodunovImpl.h
        src/vof/algo/riemann_solvers/ExactRiemannSolver.cpp
        src/vof/algo/riemann_solvers/ExactRiemannSolver.h
        src/vof/algo/timeStepCalculator/TimeStepCalculator.cpp
        src/vof/algo/timeStepCalculator/TimeStepCalculator.h
        src/vof/controller/problemInitialisers/IBaseInitialiser.h
        src/vof/controller/problemInitialisers/ShockTubeInitialiser.cpp
        src/vof/controller/problemInitialisers/ShockTubeInitialiser.h
        src/vof/controller/Controller.cpp
        src/vof/controller/Controller.h
        src/vof/data/geom/Point.cpp
        src/vof/data/geom/Point.h
        src/vof/data/geom/Point_fwd.h
        src/vof/data/geom/Vector.cpp
        src/vof/data/geom/Vector.h
        src/vof/data/geom/Vector_fwd.h
        src/vof/data/grid/kernel/StateGrid.cpp
        src/vof/data/grid/kernel/StateGrid.h
        src/vof/data/grid/EulerianGrid.cpp
        src/vof/data/grid/EulerianGrid.h
        src/vof/io/FileWriter.cpp
        src/vof/io/FileWriter.h
        src/vof/io/VTKWriter.cpp
        src/vof/io/VTKWriter.h
        src/vof/main.cpp
        src/vof/controller/problemInitialisers/NohInitialiser.cpp
        src/vof/controller/problemInitialisers/NohInitialiser.h
        src/vof/algo/boundaryConditions/ReflectingBoundaryCondition.cpp
        src/vof/algo/boundaryConditions/ReflectingBoundaryCondition.h
        src/testing/vof/simulation/SimulationTest.h
        src/vof/controller/problemInitialisers/InitialGridState.cpp
        src/vof/controller/problemInitialisers/InitialGridState.h
        src/vof/controller/problemInitialisers/DoubleMachInitialiser.cpp
        src/vof/controller/problemInitialisers/DoubleMachInitialiser.h
        src/testing/vof/simulation/ShockTubeSimulation_unittest.cpp
        src/testing/vof/simulation/NohSimulationTest.cpp
        src/testing/vof/simulation/SimulationTest.cpp
        src/vof/algo/gridUpdaters/CellUpdaterImpl.cpp
        src/vof/algo/gridUpdaters/CellUpdaterImpl.h
        src/vof/algo/boundaryConditions/BoundaryHandlerImpl.cpp
        src/vof/algo/boundaryConditions/BoundaryHandlerImpl.h
        src/vof/controller/problemInitialisers/BaseInitialiser.cpp
        src/vof/controller/problemInitialisers/BaseInitialiser.h
        src/vof/data/eos/eosProperties/IdealEquationOfStateProperties.cpp
        src/vof/data/eos/eosProperties/IdealEquationOfStateProperties.h
        src/testing/vof/simulation/DoubleMachSimulationTest.cpp src/vof/statistics/ScopedTimer.cpp src/vof/statistics/ScopedTimer.h)

add_executable(VOF ${SOURCE_FILES})
target_link_libraries(VOF gtest)
