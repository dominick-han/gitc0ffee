CXX       := clang++
MARCH     ?= -march=native
CXXFLAGS  := -std=c++20 -O3 $(MARCH) -DNDEBUG -Wall -Wextra -Wno-deprecated-declarations -mmacosx-version-min=13.0
PREFIX    ?= /usr/local
VERSION   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

OPENSSL   := $(shell brew --prefix openssl@3 2>/dev/null || echo /opt/homebrew/opt/openssl@3)
INCLUDES  := -Isrc -I$(OPENSSL)/include -Ibuild
LIBS      := -L$(OPENSSL)/lib -lcrypto -framework Metal -framework Foundation

B := build
TARGET := $(B)/gitc0ffee

SRCS_CPP := src/main.cpp src/git.cpp src/commit.cpp
SRCS_MM  := src/gpu_solver.mm
OBJS     := $(patsubst src/%.cpp,$(B)/%.o,$(SRCS_CPP)) $(patsubst src/%.mm,$(B)/%.o,$(SRCS_MM))

.PHONY: all clean test check install uninstall

all: $(TARGET)

$(B):
	mkdir -p $(B)

$(B)/shader_source.h: src/sha1.metal | $(B)
	@printf 'Embedding shader...\n'
	@{ echo '#pragma once'; echo 'static const char* kMetalShaderSource = R"METAL('; cat $<; echo ')METAL";'; } > $@

$(B)/%.o: src/%.cpp $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DVERSION='"$(VERSION)"' -c $< -o $@

$(B)/%.o: src/%.mm $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -DVERSION='"$(VERSION)"' -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $^ $(LIBS) -o $@

clean:
	rm -rf $(B)

install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/gitc0ffee

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/gitc0ffee

# --- Tests ---

$(B)/test_commit: tests/test_commit.cpp $(B)/commit.o | $(B)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(B)/commit.o -L$(OPENSSL)/lib -lcrypto -o $@

$(B)/test_gpu: tests/test_gpu.mm $(B)/commit.o $(B)/gpu_solver.o $(B)/shader_source.h | $(B)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) $< $(B)/commit.o $(B)/gpu_solver.o $(LIBS) -o $@

test: $(B)/test_commit $(B)/test_gpu $(TARGET)
	@$(B)/test_commit
	@$(B)/test_gpu
	@tests/test_e2e.sh $(TARGET)

check: test
