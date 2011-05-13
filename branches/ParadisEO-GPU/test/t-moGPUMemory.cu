/*
  <t-moGPUMemory.cu>
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

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <memory/moGPUAllocator.h>
#include <memory/moGPUDisallocator.h>
#include <memory/moGPUCopy.h>


int main() {


  std::cout << "[t-moGPUMemory] => START" << std::endl;

  int * h_data;
  int * cpy_data;
  int * d_data;
  moGPUAllocator alloc;
  moGPUDisallocator disalloc;
  moGPUCopy cpy;
  int i=0;

  //data allocation
  h_data= new int[5];
  cpy_data= new int[5];

  //test GPU data allocation
  alloc(d_data,5);

  for(i=0;i<5;i++)
    h_data[i]=i;
   
  //test default way of copy from host to device
  cpy(d_data,h_data,5);

  //test copy from device to host
  cpy(cpy_data,d_data,5,0);
  for(i=0;i<5;i++)
    assert(cpy_data[i]==i);

  for(i=0;i<5;i++)
    h_data[i]=i*2;

  //test copy from host to device
  cpy(d_data,h_data,5,1);

  //test copy from device to host
  cpy(cpy_data,d_data,5,0);
  for(i=0;i<5;i++)
    assert(cpy_data[i]==i*2);
  
  //test GPU memory disallocation
  disalloc(d_data);

  delete[] h_data;
  delete[] cpy_data;
  std::cout << "[t-moGPUMemory] => OK" << std::endl;

  return EXIT_SUCCESS;
}
 
