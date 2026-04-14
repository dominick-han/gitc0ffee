UNAME_S := $(shell uname -s)
MARCH   ?= -march=native
PREFIX  ?= /usr/local
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

B := build
TARGET := $(B)/gitc0ffee

SRCS_CPP := src/main.cpp src/git.cpp src/commit.cpp

# --- Platform-specific configuration ---

ifeq ($(UNAME_S),Darwin)
  # macOS: Metal GPU solver
  CXX      := clang++
  CXXFLAGS := -std=c++20 -O3 $(MARCH) -DNDEBUG -Wall -Wextra -Wno-deprecated-declarations -mmacosx-version-min=13.0
  OPENSSL  := $(shell brew --prefix openssl@3 2>/dev/null || echo /opt/homebrew/opt/openssl@3)
  INCLUDES := -Isrc -I$(OPENSSL)/include -Ibuild
  LIBS     := -L$(OPENSSL)/lib -lcrypto -framework Metal -framework Foundation

  SRCS_MM  := src/metal_solver.mm
  OBJS     := $(patsubst src/%.cpp,$(B)/%.o,$(SRCS_CPP)) $(patsubst src/%.mm,$(B)/%.o,$(SRCS_MM))
else
  # Linux: CPU solver with SHA-NI + threading
  CXX      := g++
  CXXFLAGS := -std=c++20 -O3 $(MARCH) -msha -msse4.1 -mssse3 -DNDEBUG -Wall -Wextra -Wno-deprecated-declarations -flto -fno-plt
  INCLUDES := -Isrc
  LIBS     := -lcrypto -lpthread

  SRCS_CPP += src/cpu_solver.cpp src/cpu_solver_avx512.cpp src/cpu_solver_sha_ni.cpp src/cpu_solver_avx2.cpp
  OBJS     := $(patsubst src/%.cpp,$(B)/%.o,$(SRCS_CPP))
endif

.PHONY: all clean test check install uninstall

all: $(TARGET)

$(B):
	mkdir -p $(B)

# --- Shader embedding (macOS only) ---

ifeq ($(UNAME_S),Darwin)
$(B)/shader_source.h: src/sha1.metal | $(B)
	@printf 'Embedding shader...\n'
	@{ echo '#pragma once'; echo 'static const char* kMetalShaderSource = R"METAL('; cat $<; echo ')METAL";'; } > $@

$(B)/%.o: src/%.mm $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -DVERSION='"$(VERSION)"' -c $< -o $@
endif

$(B)/%.o: src/%.cpp | $(B)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DVERSION='"$(VERSION)"' -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LIBS) -o $@

clean:
	rm -rf $(B)

install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/gitc0ffee

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/gitc0ffee

# --- Tests ---

$(B)/test_commit: tests/test_commit.cpp $(B)/commit.o | $(B)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(B)/commit.o $(LIBS) -o $@

ifeq ($(UNAME_S),Darwin)
$(B)/test_gpu: tests/test_gpu.mm $(B)/commit.o $(B)/metal_solver.o $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) $< $(B)/commit.o $(B)/metal_solver.o $(LIBS) -o $@

test: $(B)/test_commit $(B)/test_gpu $(TARGET)
	@$(B)/test_commit
	@$(B)/test_gpu
	@tests/test_e2e.sh $(TARGET)
else
$(B)/test_cpu: tests/test_cpu.cpp $(B)/commit.o $(B)/cpu_solver.o $(B)/cpu_solver_avx512.o $(B)/cpu_solver_sha_ni.o $(B)/cpu_solver_avx2.o | $(B)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(B)/commit.o $(B)/cpu_solver.o $(B)/cpu_solver_avx512.o $(B)/cpu_solver_sha_ni.o $(B)/cpu_solver_avx2.o $(LIBS) -o $@

test: $(B)/test_commit $(B)/test_cpu $(TARGET)
	@$(B)/test_commit
	@$(B)/test_cpu
	@tests/test_e2e.sh $(TARGET)
endif

check: test
