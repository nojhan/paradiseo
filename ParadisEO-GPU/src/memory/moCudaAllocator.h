/*
  <moCudaAllocator.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Boufaras Karima, Thé Van Luong

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

#ifndef _moCudaAllocator_H_
#define _moCudaAllocator_H_

// CUDA includes
#include <cutil.h>

/**
 *  class for allocation data on GPU global memory
 */

class moCudaAllocator {

 public:

  /**
   * Constructor
   */

  moCudaAllocator() {
  }
    template<typename T>
  /**
   *Allocate data on GPU global memory
   *@param _data the data to allocate on GPU global memory
   *@param _dataSize the size of data to allocate on GPU memory
   */
  template<typename T>
    void operator()(T* & _data, unsigned _dataSize) {

    //Allocate data in GPU memory
    CUDA_SAFE_CALL(cudaMalloc((void**) &_data, _dataSize * sizeof(T)));

    // Check if data allocation is failed
    CUT_CHECK_ERROR("Allocation of data on GPU global memory failed");

  }

  /**
   * Destructor
   */

  ~moCudaAllocator() {
  }

};

#endif
