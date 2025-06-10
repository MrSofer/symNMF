# Makefile for building the symnmf C executable

# Compiler
CC = gcc

# Compiler flags as specified in the PDF
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

# Source files for the C executable
C_SOURCES = symnmf.c

# Header files
H_HEADERS = symnmf.h

# Executable name
EXECUTABLE = symnmf

# Default target: build the executable
all: $(EXECUTABLE)

# Rule to build the executable from C source files
$(EXECUTABLE): $(C_SOURCES) $(H_HEADERS)
	$(CC) $(CFLAGS) $(C_SOURCES) -o $(EXECUTABLE) -lm # -lm links the math library

# Clean target: remove generated files
clean:
	rm -f $(EXECUTABLE) *.o *.so

# Phony targets
.PHONY: all clean
