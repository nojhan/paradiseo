/*
  <t-moOrderNeighborhood.cpp>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>
#include <paradiseo/mo/problems/bitString/moBitNeighbor.h>
#include <paradiseo/eo/ga/eoBit.h>

#include <cstdlib>
#include <cassert>

int main() {

    std::cout << "[t-moOrderNeighborhood] => START" << std::endl;

    //init sol
    eoBit<int> sol;
    sol.push_back(true);
    sol.push_back(false);
    sol.push_back(true);

    moBitNeighbor<int> neighbor;

    //verif du constructeur vide
    moOrderNeighborhood<moBitNeighbor<int> > test(3);
    assert(test.position()==0);

    //verif du hasneighbor
    assert(test.hasNeighbor(sol));

    //verif de init
    test.init(sol, neighbor);
    assert(neighbor.index()==0);
    assert(test.position()==0);

    //verif du next
    test.next(sol, neighbor);
    assert(neighbor.index()==1);
    assert(test.position()==1);

    //verif du cont
    assert(test.cont(sol));
    test.next(sol, neighbor);
    assert(!test.cont(sol));

    assert(test.className()=="moOrderNeighborhood");

    std::cout << "[t-moOrderNeighborhood] => OK" << std::endl;
    return EXIT_SUCCESS;
}
