UNAME_S := $(shell uname -s)
MARCH   ?= -march=native
PREFIX  ?= /usr/local
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

B        := build
TARGET   := $(B)/gitc0ffee
CXXFLAGS := -std=c++20 -O3 $(MARCH) -DNDEBUG -Wall -Wextra -Wno-deprecated-declarations
CPPFLAGS := -Isrc -DVERSION='"$(VERSION)"'
LIBS     := -lcrypto

SRCS := src/main.cpp src/git/git.cpp src/git/commit.cpp

ifeq ($(UNAME_S),Darwin)
  CXX      := clang++
  CXXFLAGS += -mmacosx-version-min=13.0
  OPENSSL  := $(shell brew --prefix openssl@3 2>/dev/null || echo /opt/homebrew/opt/openssl@3)
  CPPFLAGS += -I$(OPENSSL)/include -I$(B)
  LIBS     += -L$(OPENSSL)/lib -framework Metal -framework Foundation
  SRCS_MM  := src/gpu/metal_solver.mm
else
  CXX      := g++
  CXXFLAGS += -msha -msse4.1 -mssse3 -flto -fno-plt
  LIBS     += -lpthread
  SRCS     += src/cpu/solver.cpp src/cpu/avx512.cpp src/cpu/sha_ni.cpp src/cpu/avx2.cpp
endif

OBJS := $(patsubst %.cpp,$(B)/%.o,$(notdir $(SRCS))) \
        $(patsubst %.mm,$(B)/%.o,$(notdir $(SRCS_MM)))

VPATH := src src/cpu src/git src/gpu

.PHONY: all clean test check install uninstall
all: $(TARGET)

# --- Compile rules ---

$(B):
	mkdir -p $(B)

$(B)/%.o: %.cpp | $(B)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

ifeq ($(UNAME_S),Darwin)
$(B)/shader_source.h: src/gpu/sha1.metal | $(B)
	@{ echo '#pragma once'; echo 'static const char* kMetalShaderSource = R"METAL('; cat $<; echo ')METAL";'; } > $@

$(B)/%.o: %.mm $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -fobjc-arc -c $< -o $@
endif

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LIBS) -o $@

# --- Tests ---

SOLVER_OBJS = $(filter-out $(B)/main.o,$(OBJS))

$(B)/test_commit: tests/test_commit.cpp $(B)/commit.o | $(B)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< $(B)/commit.o $(LIBS) -o $@

ifeq ($(UNAME_S),Darwin)
$(B)/test_gpu: tests/test_gpu.mm $(B)/commit.o $(B)/metal_solver.o $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -fobjc-arc $< $(B)/commit.o $(B)/metal_solver.o $(LIBS) -o $@

test: $(B)/test_commit $(B)/test_gpu $(TARGET)
	@$(B)/test_commit
	@$(B)/test_gpu
	@tests/test_e2e.sh $(TARGET)
else
$(B)/test_cpu: tests/test_cpu.cpp $(SOLVER_OBJS) | $(B)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< $(SOLVER_OBJS) $(LIBS) -o $@

test: $(B)/test_commit $(B)/test_cpu $(TARGET)
	@$(B)/test_commit
	@$(B)/test_cpu
	@tests/test_e2e.sh $(TARGET)
endif

check: test

# --- Install ---

install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/gitc0ffee

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/gitc0ffee

clean:
	rm -rf $(B)
