#
# Makefile for DonutEngine
#
OS := $(shell uname)

# my laptop's OS is 'Darwin'
CC = gcc
CXX = g++
SWIGFLAGS = -python -c++ -w490 -I$cwd
XRAY_SOFTDIR = /afs/slac/g/ki/software/
#	CFLAGS = -Wall -ansi -g
	CFLAGS =  -Wall -ansi -O3  -m64 -funroll-loops -fomit-frame-pointer -ffast-math -mfpmath=sse -msse2 -mtune=native -fPIC
	LD = gcc
	LDFLAGS = -bundle -flat_namespace -undefined suppress
	LIBS = -L/usr/local/lib  -lcfitsio -lfftw3 -lm -L/Library/Frameworks/EPD64.framework/Versions/7.3/lib
	INCS = -I/Users/cpd/Projects/WavefrontPSF/code/donutlib -I/usr/local/include -I/Library/Frameworks/EPD64.framework/Versions/7.3/include/python2.7 -I/Library/Frameworks/EPD64.framework/Versions/7.3/lib/python2.7/site-packages/numpy/core/include
	SW = swig


all: swig donutengine clean_clutter

donutengine: DonutEngine.cc
	$(CXX) -c $(CFLAGS) DonutEngine.cc $(INCS) -o DonutEngine.o
	$(CXX) -c $(CFLAGS) Zernike.cc $(INCS) -o Zernike.o
	$(CXX) -c $(CFLAGS) FFTWClass.cc $(INCS) -o FFTWClass.o
	$(CXX)  $(CFLAGS) $(INCS) -c -o DonutEngineWrap.o DonutEngineWrap.cxx 
	$(LD) $(LDFLAGS) -o _donutengine.so  DonutEngineWrap.o DonutEngine.o Zernike.o FFTWClass.o  $(LIBS)

swig:
	$(SW) $(SWIGFLAGS) -o DonutEngineWrap.cxx DonutEngine.i

clean:
	rm -f *.o
	rm -f *.cxx
	rm -f *.pyc
	rm -f *.so

clean_clutter:
	rm -f *.o
	rm -f *.cxx
	rm -f *.pyc
