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

SRC_DIR := src
OBJ_DIR := build

EXE := ParticleSim
SRC := $(wildcard $(SRC_DIR)/*.cpp)
SRCCUDA := $(wildcard $(SRC_DIR)/*.cu)

CPPFLAGS := -Iinclude -MMD -MP
CXXFLAGS := -g -Wall -fopenmp -march=native
LDFLAGS := -fopenmp
LDLIBS := #-lm

# If using GPU
NVCC = nvcc
CUDAFLAGS = -g -Xcompiler -Wall,-fopenmp
LDFLAGS := -Xcompiler $(LDFLAGS)

ifeq ($(OS_NAME),windows)
	EXE += .exe
	SRC := $(filter-out %_linux.cpp, $(SRC))
	LDFLAGS += -lgdi32
else
	SRC := $(filter-out %_win.cpp, $(SRC))
	LDFLAGS += -lX11
endif

OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJCUDA := $(SRCCUDA:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS := $(OBJ) $(OBJCUDA)

.PHONY: all clean os cxx rm

all: $(EXE)

$(EXE): $(OBJS)
	$(NVCC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(OBJ_DIR)

os:
	@echo $(OS_NAME) $(ARCH)

-include $(OBJ:.o=.d)
-include $(OBJCUDA:.o=.d)
