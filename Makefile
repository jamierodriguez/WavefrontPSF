


all: 



## # --------------------------------------------------
## # if there is no Makefile.in then use the template
## # --------------------------------------------------
## ifneq ($(strip $(MAKEFILE_IN)),)
## # use value of MAKEFILE_IN if provided on the command line
## else ifeq ($(shell test -e Makefile.in && echo 1), 1)
## MAKEFILE_IN = Makefile.in
## else
## MAKEFILE_IN = Makefile.in.template
## endif
## include $(MAKEFILE_IN)
## 
## 
## SRC = $(wildcard *.c)
## OBJ = $(SRC:.c=.o)
## DEP = $(SRC:.c=.dep)
## EXE = cake
## 
## 
## ifeq ($(HAVE_HDF5), 1)
## INC += -I$(HDF5_HOME)/include
## LIB += -L$(HDF5_HOME)/lib -lhdf5
## DEF += -DCAKE_HAVE_HDF5
## endif
## 
## 
## default : $(EXE)
## 
## 
## show :
## 	@echo SRC=$(SRC)
## 	@echo OBJ=$(OBJ)
## 	@echo DEP=$(DEP)
## 
## 
## %.o : %.c $(MAKEFILE_IN)
## 	$(CC) -MM $< > $(<:.c=.dep)
## 	$(CC) $(CFLAGS) -o $@ $< $(INC) $(DEF) -c
## 
## 
## cake : $(OBJ)
## 	$(CC) $(CFLAGS) -o $@ $^ $(LIB)
## 
## 
## clean :
## 	$(RM) $(EXE) $(OBJ) $(DEP)
## 
## 
## -include *.dep


## #
## # Makefile for DonutEngine
## #
## OS := $(shell uname)
## 
## CC = gcc
## CXX = g++
## SWIGFLAGS = -python -c++ -w490 -I$cwd
## 
## #	CFLAGS = -Wall -ansi -g
## 	CFLAGS =  -Wall -ansi -O3  -m64 -funroll-loops -fomit-frame-pointer -ffast-math -mfpmath=sse -msse2 -mtune=native -fPIC
## 	LD = gcc
## 	LDFLAGS = -bundle -flat_namespace -undefined suppress
## 	LIBS = -L/usr/local/lib  -lcfitsio -lfftw3 -lm
## 	INCS = -I/Users/cpd/Dropbox/derp-ninja/DES-STANFORD/DECamOptics/donutlib -I/usr/local/include -I/Library/Frameworks/EPD64.framework/Versions/7.3/include/python2.7 -I/Library/Frameworks/EPD64.framework/Versions/7.3/lib/python2.7/site-packages/numpy/core/include
## 	SW = swig
## 
## 
## all: swig donutengine clean_clutter
## 
## donutengine: DonutEngine.cc
## 	$(CXX) -c $(CFLAGS) DonutEngine.cc $(INCS) -o DonutEngine.o
## 	$(CXX) -c $(CFLAGS) Zernike.cc $(INCS) -o Zernike.o
## 	$(CXX) -c $(CFLAGS) FFTWClass.cc $(INCS) -o FFTWClass.o
## 	$(CXX)  $(CFLAGS) $(INCS) -c -o DonutEngineWrap.o DonutEngineWrap.cxx 
## 	$(LD) $(LDFLAGS) -o _donutengine.so  DonutEngineWrap.o DonutEngine.o Zernike.o FFTWClass.o  $(LIBS)
## 
## swig:
## 	$(SW) $(SWIGFLAGS) -o DonutEngineWrap.cxx DonutEngine.i
## 
## clean:
## 	rm -f *.o
## 	rm -f *.cxx
## 	rm -f *.pyc
## 	rm -f *.so
## 
## clean_clutter:
## 	rm -f *.o
## 	rm -f *.cxx
## 	rm -f *.pyc
