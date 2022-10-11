CC=gcc
CFLAGS=-fPIC -shared
OBJECTS=strkernel.so
.PHONY: wdkernel
.PHONY: strkernel

wdkernel: strkernel.so
	@echo "\nTesting the performance of WD kernel calculation"
	python WDKernel.py

strkernel:
	@echo "\nTesting the performance of string kernel in C"
	gcc strkernel.c
	mv a.out ~
	chmod +x ~/a.out
	~/a.out

%.so: %.c
	@echo Compiling shared library
	$(CC) $(CFLAGS) $< -o $@
