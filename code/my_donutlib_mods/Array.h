//
// $Rev::                                                           $:
// $Author:: roodman                                                $:
// $LastChangedDate::                                               $:
//
//
// Array.h: A 1,2,3 Dimensional templated array class
// Array uses fftw_alloc (or fftwf_malloc) to allocate aligned memory
// This code is John Bowman's Array.h, with some modifications:
//
// Array.h:  A high-performance multi-dimensional C++ array class
// Copyright (C) 1997-2010 John C. Bowman
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */
//
// Aaron J. Roodman, SLAC National Accelerator Laboratory
//

#ifndef ARRAY_HH
#define ARRAY_HH

#include <iostream>
#include <sstream>
#include <climits>
#include <cstdlib>
#include <cerrno>
#include <complex>
#include <vector>
#include <fftw3.h>


namespace Array {

inline std::ostream& _newl(std::ostream& s) {s << '\n'; return s;}

// One dimensional array, and base class

template<class T>
void Allocate(T *&v,size_t size)
{
  void *mem = NULL;
  mem = fftw_malloc(size * sizeof(T));
  //mem = std::malloc(size * sizeof(T));
  v = (T*) mem;
}

template<class T>
class array1 {
 protected:
  T *v;
  unsigned int size;
  bool allocated;
 public:
  virtual unsigned int Size() const {return size;}
  virtual void Dimension(unsigned int nx0) {

    size=nx0;
    Allocate(v,size);
    allocated = true;
  }
  virtual void Free(){
    if (allocated) fftw_free(v);
    //if (allocated) std::free(v);
  }

  // constructors
  array1() : v(NULL), size(0), allocated(false) {}
  array1(unsigned int nx0) {
    Dimension(nx0);
  }
  array1(unsigned int nx0, T *v0): v(v0), size(nx0), allocated(false) {}
  // need a copy constructor??   array1(const array1<T>& A): size(A.size), v(A.v) {}


  virtual ~array1() {Free();}
  unsigned int Nx() const {return size;}

  T& operator [] (int ix) const {return v[ix];}
  T& operator () (int ix) const {return v[ix];}
  T* operator () () const {return v;}
  operator T* () const {return v;}

  void Load(T a) const {
    for(unsigned int i=0; i < size; i++) v[i]=a;
  }
  void Load(const T *a) const {
    for(unsigned int i=0; i < size; i++) v[i]=a[i];
  }
  /* T Min() { */
  /*   T min=v[0]; */
  /*   for(unsigned int i=1; i < size; i++) if(v[i] < min) min=v[i]; */
  /*   return min; */
  /* } */
  /* T Max() { */
  /*   T max=v[0]; */
  /*   for(unsigned int i=1; i < size; i++) if(v[i] > max) max=v[i]; */
  /*   return max; */
  /* } */

  std::istream& Input (std::istream &s) const {
    for(unsigned int i=0; i < size; i++) s >> v[i];
    return s;
  }

  void checkASize(const array1<T>& A){
    // check that this and A have same size
    if (this->size != A.Size()){
      std::cout << "Array::error arrays not the same size" << std::endl;
    }
  }

  std::vector<T> toVector(){
    std::vector<T> returnVector(size);
    for (unsigned int i=0; i<size; i++){
      returnVector[i] = v[i];
    }
    return returnVector;
  }

  array1<T>& operator = (T a) {Load(a); return *this;}
  array1<T>& operator = (const T *a) {Load(a); return *this;}
  array1<T>& operator = (const array1<T>& A) {
    this->checkASize(A);
    this->Load(A());
    return *this;
  }

  array1<T>& operator += (const array1<T>& A) {
    for(unsigned int i=0; i < size; i++) v[i] += A(i);
    return *this;
  }
  array1<T>& operator -= (const array1<T>& A) {
    for(unsigned int i=0; i < size; i++) v[i] -= A(i);
    return *this;
  }
  array1<T>& operator *= (const array1<T>& A) {
    for(unsigned int i=0; i < size; i++) v[i] *= A(i);
    return *this;
  }
  array1<T>& operator /= (const array1<T>& A) {
    for(unsigned int i=0; i < size; i++) v[i] /= A(i);
    return *this;
  }

  array1<T>& operator += (T a) {
    for(unsigned int i=0; i < size; i++) v[i] += a;
    return *this;
  }
  array1<T>& operator -= (T a) {
    for(unsigned int i=0; i < size; i++) v[i] -= a;
    return *this;
  }
  array1<T>& operator *= (T a) {
    for(unsigned int i=0; i < size; i++) v[i] *= a;
    return *this;
  }
  array1<T>& operator /= (T a) {
    T ainv=1.0/a;
    for(unsigned int i=0; i < size; i++) v[i] *= ainv;
    return *this;
  }

  T Sum() const {
    T sum(0.0);
    for(unsigned int i=0; i < size; i++) sum += v[i];
    return sum;
  }

};


// Functions

template<class T>
std::ostream& operator << (std::ostream& s, const array1<T>& A)
{
  T *p=A();
  for(unsigned int i=0; i < A.Size(); i++) {
    s << *(p++) << " ";
  }
  return s;
}

template<class T>
std::istream& operator >> (std::istream& s, const array1<T>& A)
{
  return A.Input(s);
}

// Two Dimensional array

template<class T>
class array2 : public array1<T> {
 protected:
  unsigned int nx;
  unsigned int ny;
 public:
  void Dimension(unsigned int ny0, unsigned int nx0) {
    nx = nx0;
    ny = ny0;
    this->size = nx*ny;
    array1<T>::Dimension(nx0*ny0);
  }

  array2() : nx(0), ny(0) {
    this->size = 0;
    this->v = NULL;
    this->allocated = false;
  }
  array2(unsigned int ny0, unsigned int nx0): nx(nx0), ny(ny0) {
    Dimension(nx0,ny0);
  }
  array2(unsigned int ny0, unsigned int nx0, T *v0): nx(nx0), ny(ny0) {
    this->size = nx*ny;
    this->v = v0;
    this->allocated = false;
  }

  unsigned int Nx() const {return nx;}
  unsigned int Ny() const {return ny;}

  T& operator () (int iy, int ix) const {
    return this->v[iy*nx+ix];
  }

  void AtIndex(int iy, int ix) const {
    std::cout << this->v[iy*nx+ix] << std::endl;
  }

  T& operator () (int i) const {
#ifdef NDEBUG
    // add some debug checks here!
    if (i< (int) this->size && i>=0){
#endif
      return this->v[i];
#ifdef NDEBUG
    } else {
      std::cout << "Array error: index " << i << " memory loc " << this->v << std::cout;
      return this->v[1];
    }
#endif
  }

  T* operator () () const {return this->v;}

  array1<T> operator [] (int iy) const {
    return array1<T>(nx,this->v+iy*nx);
  }

  array2<T>& operator = (T a) {this->Load(a); return *this;}
  array2<T>& operator = (T *a) {this->Load(a); return *this;}
  array2<T>& operator = (const array2<T>& A) {
    this->checkASize(A);
    this->Load(A());
    return *this;
  }

  array2<T>& operator += (const array2<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] += A(i);
    return *this;
  }
  array2<T>& operator -= (const array2<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] -= A(i);
    return *this;
  }
  array2<T>& operator *= (const array2<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= A(i);
    return *this;
  }
  array2<T>& operator /= (const array2<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] /= A(i);
    return *this;
  }

  array2<T>& operator += (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] += a;
    return *this;
  }
  array2<T>& operator -= (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] -= a;
    return *this;
  }
  array2<T>& operator *= (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= a;
    return *this;
  }
  array2<T>& operator /= (T a) {
    T ainv=1.0/a;
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= ainv;
    return *this;
  }

};

template<class T>
std::ostream& operator << (std::ostream& s, const array2<T>& A)
{
  T *p=A();
  for(unsigned int i=0; i < A.Nx(); i++) {
    for(unsigned int j=0; j < A.Ny(); j++) {
      s << *(p++) << " ";
    }
    s << _newl;
  }
  s << std::flush;
  return s;
}

template<class T>
std::istream& operator >> (std::istream& s, const array2<T>& A)
{
  return A.Input(s);
}

// Three dimensional array

template<class T>
class array3 : public array1<T> {
 protected:
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
 public:
  void Dimension(unsigned int nz0, unsigned int ny0, unsigned int nx0) {
    nz = nz0;
    ny = ny0;
    nx = nx0;
    this->size = nx*ny*nz;
    array1<T>::Dimension(nx0*ny0*nz0);
  }

  array3() : nx(0), ny(0), nz(0) {
    this->size = 0;
    this->v = NULL;
    this->allocated = false;
  }
  array3(unsigned int nx0, unsigned int ny0, unsigned int nz0): nx(nx0), ny(ny0), nz(nz0) {
    Dimension(nx0,ny0,nz0);
  }
  array3(unsigned int nz0, unsigned int ny0, unsigned int nx0, T *v0): nx(nx0), ny(ny0), nz(nz0) {
    this->size = nx*ny*nz;
    this->v = v0;
    this->allocated = false;
  }


  unsigned int Nx() const {return nx;}
  unsigned int Ny() const {return ny;}
  unsigned int Nz() const {return nz;}

  T& operator () (int iz, int iy, int ix) const {
    return this->v[(iz*ny+iy)*nx+ix];
  }
  T& operator () (int i) const {
    return this->v[i];
  }
  T* operator () () const {return this->v;}

  array3<T>& operator = (T a) {this->Load(a); return *this;}
  array3<T>& operator = (T *a) {this->Load(a); return *this;}
  array3<T>& operator = (const array3<T>& A) {
    this->checkASize(A);
    this->Load(A());
    return *this;
  }

  array2<T> operator [] (int iz) const {
    return array2<T>(ny,nx,this->v+iz*ny*nx);
  }


  array3<T>& operator += (array3<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] += A(i);
    return *this;
  }
  array3<T>& operator -= (array3<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] -= A(i);
    return *this;
  }
  array3<T>& operator *= (array3<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= A(i);
    return *this;
  }
  array3<T>& operator /= (array3<T>& A) {
    checkASize(A);
    for(unsigned int i=0; i < this->size; i++) this->v[i] /= A(i);
    return *this;
  }

  array3<T>& operator += (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] += a;
    return *this;
  }
  array3<T>& operator -= (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] -= a;
    return *this;
  }
  array3<T>& operator *= (T a) {
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= a;
    return *this;
  }
  array3<T>& operator /= (T a) {
    T ainv=1.0/a;
    for(unsigned int i=0; i < this->size; i++) this->v[i] *= ainv;
    return *this;
  }


};

template<class T>
std::ostream& operator << (std::ostream& s, const array3<T>& A)
{
  T *p=A();
  for(unsigned int i=0; i < A.Nx(); i++) {
    for(unsigned int j=0; j < A.Ny(); j++) {
      for(unsigned int k=0; k < A.Nz(); k++) {
	s << *(p++) << " ";
      }
      s << _newl;
    }
    s << _newl;
  }
  s << std::flush;
  return s;
}

template<class T>
std::istream& operator >> (std::istream& s, const array3<T>& A)
{
  return A.Input(s);
}

}
#endif
