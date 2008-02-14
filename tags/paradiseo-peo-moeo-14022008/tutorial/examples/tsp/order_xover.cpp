/*
* <order_xover.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
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

#include "order_xover.h"

void OrderXover :: cross (const Route & __par1, const Route & __par2, Route & __child)
{

  unsigned cut2 = 1 + rng.random (numNodes) ;
  unsigned cut1 = rng.random (cut2);
  unsigned l = 0;

  /* To store vertices that have already been crossed */
  std :: vector <bool> v (numNodes, false);

  /* Copy of the left partial route of the first parent */
  for (unsigned i = cut1 ; i < cut2 ; i ++)
    {
      __child [l ++] = __par1 [i] ;
      v [__par1 [i]] = true ;
    }

  /* Searching the vertex of the second path, that ended the previous first one */
  unsigned from = 0 ;
  for (unsigned i = 0; i < numNodes; i ++)
    if (__par2 [i] == __child [cut2 - 1])
      {
        from = i ;
        break ;
      }

  /* Selecting a direction (Left or Right) */
  char direct = rng.flip () ? 1 : -1 ;

  for (unsigned i = 0; i < numNodes + 1; i ++)
    {
      unsigned bidule = (direct * i + from + numNodes) % numNodes;
      if (! v [__par2 [bidule]])
        {
          __child [l ++] = __par2 [bidule] ;
          v [__par2 [bidule]] = true ;
        }
    }
}

bool OrderXover :: operator () (Route & __route1, Route & __route2)
{

  // Init. copy
  Route par [2] ;
  par [0] = __route1 ;
  par [1] = __route2 ;

  cross (par [0], par [1], __route1) ;
  cross (par [1], par [0], __route2) ;

  __route1.invalidate () ;
  __route2.invalidate () ;

  return true ;
}
