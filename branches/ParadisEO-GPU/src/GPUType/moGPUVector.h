/*
  <moGPUVector.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Karima Boufaras, Thé Van LUONG

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moGPUVector_H_
#define __moGPUVector_H_

#include <eo>

/**
 * Implementation of a GPU solution representation.
 */

template<class ElemT, class Fitness>

  class moGPUVector: public EO<Fitness> {

 public:

    /**
     * Define vector type corresponding to Solution
     */
	typedef ElemT ElemType;

	/**
	 * Default constructor.
	 */

	moGPUVector() :
		N(0) {
	}

    /**
     *Constructor.
     *@param _neighborhoodSize The neighborhood size.
     */

	 moGPUVector(unsigned _neighborhoodSize) :
		N(_neighborhoodSize) {

		vect = new ElemType[N];

	}

    /**
     *Copy Constructor
     *@param _vector The vector passed to the function to determine the new content.
     */

    moGPUVector(const moGPUVector & _vector) {

      N = _vector.N;
      vect = new ElemType[N];
      for (unsigned i = 0; i < N; i++)
	vect[i] = _vector.vect[i];
      if (!(_vector.invalid()))
	fitness(_vector.fitness());
      else
	(*this).invalidate();
    }

    /**
     * Destructor.
     */

    ~moGPUVector() {
      if (N >= 1)
	delete[] vect;
    }

    /**
     *How to fill the solution vector.
     */

    virtual void create() =0;

    /**
     *Assignment operator
     *@param _vector The vector passed to the function to determine the new content.
     *@return a new vector.
     */

    moGPUVector& operator=(const moGPUVector & _vector) {

      if (!(N == _vector.N)) {
	delete[] vect;
	N = _vector.N;
	vect = new ElemType[N];
      }
      for (unsigned i = 0; i < N; i++)
	vect[i] = _vector.vect[i];
      if (!(_vector.invalid()))
	fitness(_vector.fitness());
      else
	(*this).invalidate();
      return (*this);

    }

    /**
     *An accessor read only on the i'th element of the vector (function inline can be called from host or device).
     *@param _i The i'th element of vector.
     *@return The i'th element of the vector for read only
     */

    inline __host__ __device__ const ElemType & operator[](unsigned _i) const {
      if(_i<N)
	return vect[_i];
    }

    /**
     *An accessor read-write on the i'th element of the vector(function inline can be called from host or device).
     *@param _i The i'th element of the vector.
     *@return The i'th element of the vector for read-write
     */

    inline __host__ __device__ ElemType & operator[](unsigned _i) {
      if(_i<N)
	return vect[_i];
    }

    /**
     *Function inline to get the size of vector, called from host and device.
     *@return The vector size's
     */

    inline __host__ __device__ unsigned size() {

      return N;

    }

    /**
     *Function inline to set the size of vector, called from host and device.
     *@param _size the vector size
     */

    virtual inline __host__ void setSize(unsigned _size)=0;

    /**
     * Write object. Called printOn since it prints the object _on_ a stream.
     * @param _os A std::ostream.
     */

    virtual void printOn(std::ostream& os) const=0;

 protected:

    ElemType * vect;
    unsigned N;

};

#endif
