CC = g++
LD = g++
CFLAGS = -g -Wall -fopenmp -march=native
LDFLAGS = -lgdi32 -fopenmp
# -lpthread
RM = del -fR
OBJS = Program.o Vector.o Window.o
TARGET = Program.exe

all: $(TARGET)
	$(RM) *.o

$(TARGET): $(OBJS)
	$(LD) -o $(TARGET) $(OBJS) $(LDFLAGS)

Vector.o: Vector.cpp Vector.h
	$(CC) $(CFLAGS) -c Vector.cpp

Window.o: Window.cpp Window.h Vector.h
	$(CC) $(CFLAGS) -c Window.cpp

Program.o: Program.cpp Vector.h Window.h
	$(CC) $(CFLAGS) -c Program.cpp

clean:
	$(RM) $(TARGET) *.o
