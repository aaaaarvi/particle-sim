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

# OS specific settings
ifeq ($(OS_NAME),windows)
	CC = g++
	LD = g++
	CFLAGS = -g -Wall -fopenmp -march=native
	LDFLAGS = -lgdi32 -fopenmp
	# -lpthread
	RM = del -fR
	OBJS = Program.o Vector.o Window.o
	TARGET = Program.exe
	WINDOW_SRC = Window.cpp
	WINDOW_HDR = Window.h
else
	CC = g++
	LD = g++
	CFLAGS = -g -Wall -fopenmp -march=native
	LDFLAGS = -lX11 -fopenmp
	# -lpthread
	RM = rm -f
	OBJS = Program.o Vector.o WindowLinux.o
	TARGET = Program
	WINDOW_SRC = WindowLinux.cpp
	WINDOW_HDR = WindowLinux.h
endif
# TODO: OSX

all: $(TARGET)
	$(RM) *.o

$(TARGET): $(OBJS)
	$(LD) -o $(TARGET) $(OBJS) $(LDFLAGS)

Vector.o: Vector.cpp Vector.h
	$(CC) $(CFLAGS) -c Vector.cpp

Window.o: Window.cpp Window.h Vector.h
	$(CC) $(CFLAGS) -c Window.cpp

WindowLinux.o: WindowLinux.cpp WindowLinux.h Vector.h
	$(CC) $(CFLAGS) -c WindowLinux.cpp

Program.o: Program.cpp Vector.h $(WINDOW_HDR)
	$(CC) $(CFLAGS) -c Program.cpp

clean:
	$(RM) $(TARGET) *.o

os:
	@echo $(OS_NAME) $(ARCH)
