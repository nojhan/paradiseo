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
    unsigned int i, dimension;

    std::string string_read, buffer;

    std :: ifstream file (__fileName) ;

    std :: cout << ">> Loading [" << __fileName << "]" << std :: endl ;

    if (file)
      {
	// Read NAME:
	file >> string_read;
	if (string_read.compare("NAME:")!=0)
	  {
	    std::cout << "ERROR: \'NAME:\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }
	// Read instance name
	file >> string_read;
	std::cout << "\t Instance Name = " << string_read << std::endl;
	// Read TYPE:
	file >> string_read;
	if (string_read.compare("TYPE:")!=0)
	  {
	    std::cout << "ERROR: \'TYPE:\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }
	// Read instance type;
	file >> string_read;
	std::cout << "\t Instance type = " << string_read << std::endl;
	if (string_read.compare("TSP")!=0)
	  {
	    std::cout << "ERROR: only TSP type instance can be loaded" << std::endl;
	    exit(1);
	  }
	// Read COMMENT:
	file >> string_read;
	if (string_read.compare("COMMENT:")!=0)
	  {
	    std::cout << "ERROR: \'COMMENT:\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }
	// Read comments
	std::cout << "\t Instance comments = ";
	file >> string_read;
	buffer = string_read+"_first";
	while((string_read.compare("DIMENSION:")!=0) && (string_read.compare(buffer)!=0))
	  {
	    if(string_read.compare("COMMENT:")!=0)
	      {
		std::cout << string_read << " ";
	      }
	    else
	      {
		std::cout << std::endl << "\t                     ";
	      }
	    buffer = string_read;
	    file >> string_read;
	  }

	std::cout << std::endl;

	// Read dimension;
	file >> dimension ;
        std::cout << "\t Instance dimension = " << dimension << std::endl;
	vectCoord.resize (dimension) ;

	// Read EDGE_WEIGHT_TYPE
	file >> string_read;
	if (string_read.compare("EDGE_WEIGHT_TYPE:")!=0)
	  {
	    std::cout << "ERROR: \'EDGE_WEIGHT_TYPE:\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }

	// Read edge weight type
	file >> string_read;
	std::cout << "\t Instance edge weight type = " << string_read << std::endl;
	if (string_read.compare("EUC_2D")!=0)
	  {
	    std::cout << "ERROR: only EUC_2D edge weight type instance can be loaded" << std::endl;
	    exit(1);
	  }

	// Read NODE_COORD_SECTION
	file >> string_read;
	if (string_read.compare("NODE_COORD_SECTION")!=0)
	  {
	    std::cout << "ERROR: \'NODE_COORD_SECTION\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }

	// Read coordonates.
	for(i=0;i<dimension;i++)
	  {
	    // Read index
	    file >> string_read;
	    //Read Coordinate
	    file >> vectCoord [i].first >> vectCoord [i].second ;
	  }

	// Read EOF
	file >> string_read;
	if(string_read.compare("EOF")!=0)
	  {
	    std::cout << "ERROR: \'EOF\' espected, \'" << string_read << "\' found" << std::endl;
	    exit(1);
	  }

	std::cout << std::endl;
	
	file.close () ;

        computeDistances () ;
      }
    else
      {

        std :: cout << __fileName << " does not exist !!!" << std :: endl ;
        // Bye !!!
        exit (1) ;
      }
  }

  float distance (unsigned int __from, unsigned int __to)
  {
    return (float)(dist [__from] [__to]) ;
  }
}


