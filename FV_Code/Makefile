# the compiler: g++ for the C++ program being generated
CC = g++

# compiler flags:
INC = -I/Users/Rohan/Programming/FV_Code/src
VTKLIBS = -I/usr/local/Cellar/vtk/8.1.2/include/vtk-8.1 -L/usr/local/Cellar/vtk/8.1.2/lib
#  $(wildcard /usr/local/Cellar/vtk/8.1.2/lib/*.dylib)

CFLAGS = -std=c++11  $(INC) $(VTKLIBS) -lgtest -o fv_code_exec


# build files
TARGET = $(wildcard *.cpp) $(wildcard */*.cpp) $(wildcard */*/*.cpp) $(wildcard */*/*/*.cpp) $(wildcard */*/*/*/*.cpp) $(wildcard */*/*/*/*/*.cpp)

target: $(TARGET) 
	$(CC) $(CFLAGS) $(TARGET)

clean: 
	$(RM) $(TARGET)
