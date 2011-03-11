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

#include "graph.h"

using std::cout;
using std::endl;

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
            dist [i] [j] = dist [j] [i] = (unsigned int) (sqrt ((float) (distX * distX + distY * distY)) + 0.5) ;
          }
      }
  }

  void load (const char * _fileName)
  {
    unsigned int i, dimension;

    std::string string_read, buffer;

    std :: ifstream file (_fileName) ;

    cout << endl << "\tLoading [" << _fileName << "]" << endl << endl;

    if( file.is_open() )
      {
	// Read NAME:
	file >> string_read;
	if (string_read.compare("NAME:")!=0)
	  {
	    cout << "ERROR: \'NAME:\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
	  }
	// Read instance name
	file >> string_read;
	cout << "\t\tInstance Name             = " << string_read << endl;
	// Read TYPE:
	file >> string_read;
	if (string_read.compare("TYPE:")!=0)
	  {
	    cout << "ERROR: \'TYPE:\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
	  }
	// Read instance type;
	file >> string_read;
	cout << "\t\tInstance type             = " << string_read << endl;
	if (string_read.compare("TSP")!=0)
	  {
	    cout << "ERROR: only TSP type instance can be loaded" << endl;
	    exit(EXIT_FAILURE);
	  }
	// Read COMMENT:
	file >> string_read;
	if (string_read.compare("COMMENT:")!=0)
	  {
	    cout << "ERROR: \'COMMENT:\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
	  }
	// Read comments
	cout << "\t\tInstance comments         = ";
	file >> string_read;
	buffer = string_read+"_first";
	while((string_read.compare("DIMENSION:")!=0) && (string_read.compare(buffer)!=0))
	  {
	    if(string_read.compare("COMMENT:")!=0)
	      {
		cout << string_read << " ";
	      }
	    else
	      {
		cout << endl << "\t                     ";
	      }
	    buffer = string_read;
	    file >> string_read;
	  }

	cout << endl;

	// Read dimension;
	file >> dimension ;
        cout << "\t\tInstance dimension        = " << dimension << endl;
	vectCoord.resize (dimension) ;

	// Read EDGE_WEIGHT_TYPE
	file >> string_read;
	if (string_read.compare("EDGE_WEIGHT_TYPE:")!=0)
	  {
	    cout << "ERROR: \'EDGE_WEIGHT_TYPE:\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
	  }

	// Read edge weight type
	file >> string_read;
	cout << "\t\tInstance edge weight type = " << string_read << endl;
	if (string_read.compare("EUC_2D")!=0)
	  {
	    cout << "ERROR: only EUC_2D edge weight type instance can be loaded" << endl;
	    exit(EXIT_FAILURE);
	  }

	// Read NODE_COORD_SECTION
	file >> string_read;
	if (string_read.compare("NODE_COORD_SECTION")!=0)
	  {
	    cout << "ERROR: \'NODE_COORD_SECTION\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
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
	    cout << "ERROR: \'EOF\' espected, \'" << string_read << "\' found" << endl;
	    exit(EXIT_FAILURE);
	  }

	cout << endl;
	
	file.close () ;

        computeDistances () ;
      }
    else
      {
        cout << _fileName << " does not exist !!!" << endl ;
        exit(EXIT_FAILURE) ;
      }
  }

  float distance (unsigned int _from, unsigned int _to)
  {
    return (float)(dist [_from] [_to]) ;
  }
}


