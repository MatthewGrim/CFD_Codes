# the compiler: g++ for the C++ program being generated
CC = g++

# compiler flags:
CFLAGS = -Wall -Wextra -std=c++11 -lgtest

# build files
TARGET = main.cpp interpolator/AirfoilInterpolator.cpp testing/interpolator/AirfoilInterpolator_unittest.cpp

target: $(TARGET)
	$(CC) $(CFLAGS) $(TARGET)

clean:
	$(RM) $(TARGET)
