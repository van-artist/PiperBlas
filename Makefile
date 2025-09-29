CXX := g++
OPENBLAS_HOME ?= /opt/homebrew/opt/openblas

CXXFLAGS := -O3 -Isrc -I$(OPENBLAS_HOME)/include
LDFLAGS  := -L$(OPENBLAS_HOME)/lib -lopenblas -Wl,-rpath,$(OPENBLAS_HOME)/lib

SRC_ALL  := $(wildcard src/*.cpp) $(wildcard src/*.c)
APP_SRC  := src/main.cpp
CORE_SRC := $(filter-out $(APP_SRC), $(SRC_ALL))

CORE_OBJ := $(patsubst src/%,build/%, $(CORE_SRC:.cpp=.o))
CORE_OBJ := $(patsubst src/%,build/%, $(CORE_OBJ:.c=.o))
APP_OBJ  := build/main.o

TEST_SRC := $(wildcard tests/*.cpp) $(wildcard tests/*.c)
TEST_OBJ := $(patsubst tests/%.cpp,build/tests/%.o,$(TEST_SRC))
TEST_OBJ := $(patsubst tests/%.c,build/tests/%.o,$(TEST_OBJ))
TEST_BIN := $(patsubst build/tests/%.o,build/tests/%,$(TEST_OBJ))

TARGET := build/piper-blas

.PHONY: all clean tests test run-tests build_dir

all: $(TARGET)

build_dir:
	mkdir -p build build/tests

$(TARGET): $(CORE_OBJ) $(APP_OBJ) | build_dir
	$(CXX) -o $@ $^ $(LDFLAGS)

$(APP_OBJ): $(APP_SRC) | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cpp | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.c | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/tests/%.o: tests/%.cpp | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/tests/%.o: tests/%.c | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/tests/%: build/tests/%.o $(CORE_OBJ) | build_dir
	$(CXX) -o $@ $^ $(LDFLAGS)

tests: $(TEST_BIN)
test: tests

run-tests: tests
	@set -e; for t in $(TEST_BIN); do echo "==> $$t"; "$$t"; done

clean:
	rm -rf build