// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "graph.cpp"

// (c) OPAC Team, LIFL, 2003

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
*/

#include <fstream>
#include <iostream>
#include <math.h>

#include "graph.h"

namespace Graph {

  static std :: vector <std :: pair <double, double> > vectCoord ; // Coordinates
  
  static std :: vector <std :: vector <unsigned> > dist ; // Distances Mat.

  unsigned size () {
    
    return dist.size () ;
  }

  void computeDistances () {
  
    // Dim.
    unsigned numCities = vectCoord.size () ;
    dist.resize (numCities) ;
    for (unsigned i = 0 ; i < dist.size () ; i ++)
      dist [i].resize (numCities) ;
    
    // Computations.
    for (unsigned i = 0 ; i < dist.size () ; i ++)
      for (unsigned j = i + 1 ; j < dist.size () ; j ++) {
	double distX = vectCoord [i].first - vectCoord [j].first ;
	double distY = vectCoord [i].second - vectCoord [j].second ;
	dist [i] [j] = dist [j] [i] = (unsigned) (sqrt ((float) (distX * distX + distY * distY)) + 0.5) ;
      }
  }

  void load (const char * __fileName) {
  
    std :: ifstream f (__fileName) ;
  
    std :: cout << ">> Loading [" << __fileName << "]" << std :: endl ;
    
    if (f) {
    
      unsigned num_vert ; 
      
      f >> num_vert ;
      vectCoord.resize (num_vert) ;

      for (unsigned i = 0 ; i < num_vert ; i ++)	
	f >> vectCoord [i].first >> vectCoord [i].second ;
                  
      f.close () ;
      
      computeDistances () ;
    }
    else {
      
      std :: cout << __fileName << " doesn't exist !!!" << std :: endl ;
      // Bye !!!
      exit (1) ;
    }
  }

  float distance (unsigned __from, unsigned __to) {
    
    return dist [__from] [__to] ;
  }
}


