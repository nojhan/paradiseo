/* <two_opt_tabu_list.cpp>  
 *
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Sebastien CAHON
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
 */

#include "two_opt_tabu_list.h"
#include "graph.h"

#define TABU_LENGTH 10

void TwoOptTabuList :: init () 
{
  // Size (eventually)
  tabu_span.resize (Graph :: size ()) ;
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      tabu_span [i].resize (Graph :: size ()) ;  
    }

  // Clear
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      for (unsigned int j = 0 ; j < tabu_span [i].size () ; j ++)
	{
	  tabu_span [i] [j] = 0 ;
	}
    }
}

bool TwoOptTabuList :: operator () (const TwoOpt & __move, const Route & __sol) 
{
  return tabu_span [__move.first] [__move.second] > 0 ;
}

void TwoOptTabuList :: add (const TwoOpt & __move, const Route & __sol) 
{
  tabu_span [__move.first] [__move.second] = tabu_span [__move.second] [__move.first] = TABU_LENGTH ;
}

void TwoOptTabuList :: update () 
{
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      for (unsigned int j = 0 ; j < tabu_span [i].size () ; j ++)
	{
	  if (tabu_span [i] [j] > 0)
	    {
	      tabu_span [i] [j] -- ;
	    }
	}
    }
}
