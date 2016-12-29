# the compiler: g++ for the C++ program being generated
CC = g++

# compiler flags:
CFLAGS = -Wall -Wextra -std=c++11 -lgtest -o unittest

# build files
TARGET = main.cpp interpolator/AirfoilInterpolator.cpp testing/interpolator/AirfoilInterpolator_unittest.cpp solver/BEMSolver.cpp testing/solver/BEMSolver_unittest.cpp

target: $(TARGET)
	$(CC) $(CFLAGS) $(TARGET)

clean:
	$(RM) $(TARGET)
