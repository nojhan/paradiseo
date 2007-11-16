/*
* <node.cpp>
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

#include <math.h>
#include <values.h>

#include "node.h"

unsigned numNodes; /* Number of nodes */

//static unsigned * * dist; /* Square matrix of distances */

double * X_coord, * Y_coord;

double X_min = MAXDOUBLE, X_max = MINDOUBLE, Y_min = MAXDOUBLE, Y_max = MINDOUBLE;

void loadNodes (FILE * __f)
{

  /* Coord */

  X_coord = new double [numNodes];

  Y_coord = new double [numNodes];

  unsigned num;

  for (unsigned i = 0; i < numNodes; i ++)
    {

      fscanf (__f, "%u%lf%lf", & num, X_coord + i, Y_coord + i);

      if (X_coord [i] < X_min)
        X_min = X_coord [i];
      if (X_coord [i] > X_max)
        X_max = X_coord [i];
      if (Y_coord [i] < Y_min)
        Y_min = Y_coord [i];
      if (Y_coord [i] > Y_max)
        Y_max = Y_coord [i];
    }

  /* Allocation */
  /*
  dist = new unsigned * [numNodes];

  for (unsigned i = 0; i < numNodes; i ++)
    dist [i] = new unsigned [numNodes];
  */
  /* Computation of the distances */

  /*
  for (unsigned i = 0; i < numNodes; i ++) {

    dist [i] [i] = 0;

    for (unsigned j = 0; j < numNodes; j ++) {
    
      double dx = X_coord [i] - X_coord [j], dy = Y_coord [i] - Y_coord [j];
    
      dist [i] [j] = dist [j] [i] = (unsigned) (sqrt (dx * dx + dy * dy) + 0.5) ;
    }
    }*/
}

unsigned distance (Node __from, Node __to)
{

  //  return dist [__from] [__to];

  double dx = X_coord [__from] - X_coord [__to], dy = Y_coord [__from] - Y_coord [__to];

  return (unsigned) (sqrt (dx * dx + dy * dy) + 0.5) ;
}

