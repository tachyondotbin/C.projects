# Variables
CC = mpicc
# Added -O3 (optimization) and -march=native (CPU specific)
CFLAGS = -I/usr/include/SDL2 -O3 -march=native
LDLIBS = -lSDL2 -lm
TARGET = bh_sim
NP = 4

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): blk.c
	$(CC) $(CFLAGS) blk.c -o $(TARGET) $(LDLIBS)

# Rule to build and run
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

# Rule to remove the compiled binary
clean:
	rm -f $(TARGET)

.PHONY: all clean run

