/*
* <graph.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Jean-Charles Boisson
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

#include <fstream>
#include <iostream>
#include <math.h>

#include "graph.h"

namespace Graph
  {

  static std :: vector <std :: pair <double, double> > vectCoord ; // Coordinates

  static std :: vector <std :: vector <unsigned int> > dist ; // Distances Mat.

  unsigned size ()
  {
    return dist.size () ;
  }

  void computeDistances ()
  {

    // Dim.
    unsigned int numCities = vectCoord.size () ;
    dist.resize (numCities) ;
    for (unsigned int i = 0 ; i < dist.size () ; i ++)
      {
        dist [i].resize (numCities) ;
      }

    // Computations.
    for (unsigned int i = 0 ; i < dist.size () ; i ++)
      {
        for (unsigned int j = i + 1 ; j < dist.size () ; j ++)
          {
            double distX = (double)(vectCoord [i].first - vectCoord [j].first) ;
            double distY = (double)(vectCoord [i].second - vectCoord [j].second) ;
            dist [i] [j] = dist [j] [i] = (unsigned) (sqrt ((float) (distX * distX + distY * distY)) + 0.5) ;
          }
      }
  }

  void load (const char * __fileName)
  {

    std :: ifstream f (__fileName) ;

    std :: cout << ">> Loading [" << __fileName << "]" << std :: endl ;

    if (f)
      {
        unsigned int num_vert ;

        f >> num_vert ;
        vectCoord.resize (num_vert) ;

        for (unsigned int i = 0 ; i < num_vert ; i ++)
          {
            f >> vectCoord [i].first >> vectCoord [i].second ;
          }

        f.close () ;

        computeDistances () ;
      }
    else
      {

        std :: cout << __fileName << " doesn't exist !!!" << std :: endl ;
        // Bye !!!
        exit (1) ;
      }
  }

  float distance (unsigned int __from, unsigned int __to)
  {
    return (float)(dist [__from] [__to]) ;
  }
}


