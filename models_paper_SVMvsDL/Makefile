CC=gcc
CFLAGS=-O2 -fPIC -shared
OBJECTS=strkernel.so
.PHONY: wdkernel
.PHONY: strkernel
.PHONY: lib

lib: strkernel.c
	@echo Compiling shared library
	$(CC) $(CFLAGS) strkernel.c -o strkernel.so

wdkernel: strkernel.so
	@echo "\nTesting the performance of WD kernel calculation"
	python WDKernel.py

strkernel:
	@echo "\nTesting the performance of string kernel in C"
	gcc -O2 strkernel.c
	mv a.out ~
	chmod +x ~/a.out
	~/a.out

%.so: %.c
	@echo Compiling shared library
	$(CC) $(CFLAGS) $< -o $@
