# Get OS and architecture
ifeq ($(OS),Windows_NT)
	OS_NAME := windows
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		ARCH := amd64
	else ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		ARCH := x86
	else
		ARCH := $(PROCESSOR_ARCHITECTURE)
	endif
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OS_NAME := linux
	else ifeq ($(UNAME_S),Darwin)
		OS_NAME := osx
	else
		OS_NAME := $(UNAME_S)
	endif
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		ARCH := amd64
	else ifneq ($(filter %86,$(UNAME_P)),)
		ARCH := x86
	else ifneq ($(filter arm%,$(UNAME_P)),)
		ARCH := arm
	else
		ARCH := $(UNAME_P)
	endif
endif

# Determine build mode (cpu or gpu)
ifneq ($(filter cpu,$(MAKECMDGOALS)),)
	MODE := cpu
else
	MODE := gpu
endif

# Directories and files
SRC_DIR := src
OBJ_DIR := build
SRC := $(wildcard $(SRC_DIR)/*.cpp)
SRCCUDA := $(wildcard $(SRC_DIR)/*.cu)
EXE := ParticleSim

# Compiler and flags
LD := g++
CPPFLAGS := -Iinclude -MMD -MP
CXXFLAGS := -g -Wall -fopenmp -march=native
LDFLAGS := -fopenmp
LDLIBS := #-lm

# Compile for GPU
NVCC := nvcc
CUDAFLAGS := -g -Xcompiler -Wall,-fopenmp
ifeq ($(MODE),gpu)
	SRC := $(filter-out %_cpu.cpp, $(SRC))
	LD := $(NVCC)
	LDFLAGS := -Xcompiler $(LDFLAGS)
	CXXFLAGS += -DUSE_GPU
endif

# OS-specific settings
ifeq ($(OS_NAME),windows)
	EXE := $(EXE).exe
	SRC := $(filter-out %_linux.cpp, $(SRC))
	LDLIBS += -lgdi32
	MKDIR = mkdir
	RM = rmdir /s /q
else
	SRC := $(filter-out %_win.cpp, $(SRC))
	LDLIBS += -lX11
	MKDIR = mkdir -p
	RM = rm -f -rv
endif

# Object files
OBJS := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
ifeq ($(MODE),gpu)
	OBJS += $(SRCCUDA:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
endif

.PHONY: all cpu gpu clean help os

all: $(EXE)
cpu: $(EXE)
gpu: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -c $< -o $@

$(OBJ_DIR):
	$(MKDIR) $@

clean:
	@$(RM) $(OBJ_DIR)

help:
	@echo "Makefile for ParticleSim"
	@echo "Usage:"
	@echo "  make [cpu|gpu]   Build the project (default is 'gpu')"
	@echo "  make clean       Clean up build files"
	@echo "  make help        Show this help message"
	@echo "  make os          Show detected OS and architecture"

os:
	@echo $(OS_NAME) $(ARCH)

-include $(OBJS:.o=.d)
