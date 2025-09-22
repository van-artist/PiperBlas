CXX = g++
CXXFLAGS = -I./include -O2
LDFLAGS = -lblas

SRC := $(wildcard src/*.cpp) $(wildcard src/*.c)
OBJ := $(patsubst src/%,build/%, $(SRC:.cpp=.o))
OBJ := $(patsubst src/%,build/%, $(OBJ:.c=.o))

TARGET = build/piper-blas

all: build_dir $(TARGET)

build_dir:
	mkdir -p build

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.cpp | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.c | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build