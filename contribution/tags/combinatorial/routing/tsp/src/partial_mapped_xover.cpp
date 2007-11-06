/* 
* <partial_mapped_xover.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Thomas Legrand
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include <assert.h>

#include <utils/eoRNG.h>

#include "partial_mapped_xover.h"
#include "route_valid.h"
#include "mix.h"

void PartialMappedXover :: repair (Route & __route, unsigned __cut1, unsigned __cut2) {
  
  unsigned v [__route.size ()] ; // Number of times a cities are visited ...
  
  for (unsigned i = 0 ; i < __route.size () ; i ++)
    v [i] = 0 ;
  
  for (unsigned i = 0 ; i < __route.size () ; i ++)
    v [__route [i]] ++ ;
  
  std :: vector <unsigned> vert ;

  for (unsigned i = 0 ; i < __route.size () ; i ++)
    if (! v [i])
      vert.push_back (i) ;
  
  mix (vert) ;

  for (unsigned i = 0 ; i < __route.size () ; i ++)
    if (i < __cut1 || i >= __cut2)
      if (v [__route [i]] > 1) {
	__route [i] = vert.back () ;
	vert.pop_back () ;
      }
}

bool PartialMappedXover :: operator () (Route & __route1, Route & __route2) {
    
  unsigned cut1 = rng.random (__route1.size ()), cut2 = rng.random (__route2.size ()) ;
  
  if (cut2 < cut1)
    std :: swap (cut1, cut2) ;
  
  // Between the cuts
  for (unsigned i = cut1 ; i < cut2 ; i ++)
    std :: swap (__route1 [i], __route2 [i]) ;
  
  // Outside the cuts
  repair (__route1, cut1, cut2) ;
  repair (__route2, cut1, cut2) ;
  
  // Debug
  assert (valid (__route1)) ;
  assert (valid (__route2)) ;

  __route1.invalidate () ;
  __route2.invalidate () ;

  return true ;
}
