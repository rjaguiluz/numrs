# Makefile for numrs-c benchmarks on Windows with MSVC

CC = cl
CFLAGS = /nologo /O2 /W3 /I..\include
LDFLAGS = /nologo
LIBS = ..\target\release\numrs_c.dll.lib

all: benchmark_arrayview.exe

benchmark_arrayview.exe: benchmark_arrayview.c
	$(CC) $(CFLAGS) benchmark_arrayview.c /link $(LDFLAGS) $(LIBS)

clean:
	del /Q *.exe *.obj 2>nul

.PHONY: all clean
