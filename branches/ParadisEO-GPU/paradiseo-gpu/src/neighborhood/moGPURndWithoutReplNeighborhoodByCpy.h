/*
  <moGPURndWithoutReplNeighborhood.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Jerémie Humeau, Boufaras Karima, Thé Van LUONG
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

#ifndef __moGPURndWithoutReplNeighborhood_h
#define __moGPURndWithoutReplNeighborhood_h

#include <neighborhood/moRndWithoutReplNeighborhood.h>
#include <eval/moGPUEval.h>

/**
 * A Random without replacement Neighborhood with parallel evaluation
 */
template<class Neighbor>
class moGPURndWithoutReplNeighborhood: public moRndWithoutReplNeighborhood<Neighbor> {
 public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  using moRndWithoutReplNeighborhood<Neighbor>::neighborhoodSize;
  using moRndWithoutReplNeighborhood<Neighbor>::maxIndex;
  using moRndWithoutReplNeighborhood<Neighbor>::indexVector;
  /**
   * Constructor
   * @param _neighborhoodSize the size of the neighborhood
   * @param _eval show how to evaluate neighborhood of a solution at one time
   */
 moGPURndWithoutReplNeighborhood(unsigned int _neighborhoodSize,moGPUEval<
				  Neighbor>& _eval) :
  moRndWithoutReplNeighborhood<Neighbor> (_neighborhoodSize),eval(_eval) {
    for (unsigned int i = 0; i < neighborhoodSize; i++)
      indexVector.push_back(i);
  }

  /**
   * Initialization of the neighborhood
   * @param _solution the solution to explore
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    moRndWithoutReplNeighborhood<Neighbor>::init(_solution, _neighbor);
    //Compute all neighbors fitness at one time
    eval.neighborhoodEval(_solution,0,1);
  }

  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moGPURndWithoutReplNeighborhood";
  }

 protected:
  moGPUEval<Neighbor>& eval;
};

#endif
