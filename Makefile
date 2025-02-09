# A simple Makefile that compiles our C code into a shared library usable by Python (ctypes)

CC = gcc
CFLAGS = -O3 -fPIC
LIBNAME = compute_energy.so

all: $(LIBNAME)

$(LIBNAME): compute_energy.c compute_energy.h
	$(CC) $(CFLAGS) -shared -o $(LIBNAME) compute_energy.c

clean:
	rm -f *.o *.so
