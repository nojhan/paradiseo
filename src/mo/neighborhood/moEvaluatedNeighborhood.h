 /*
 <moEvaluatedNeighborhood.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

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

#ifndef _moEvaluatedNeighborhood_h
#define _moEvaluatedNeighborhood_h

#include "moNeighborhood.h"
#include "../eval/moNeighborhoodEvaluation.h"

/**
 * A Neighborhood for the evaluation of all neighbors
 * in one step
 *
 * It is usefull for example in a double incremental evaluation (QAP, UBQP problems)
 * This class is used in combinaison with the class moNeighborhoodEvaluation
 */
template<class Neighbor>
class moEvaluatedNeighborhood: public moNeighborhood<Neighbor> {
public:

  /**
   * Define type of a solution corresponding to Neighbor
   */
  typedef typename Neighbor::EOT EOT;

  /**
   * Constructor
   * @param _originalNeighborhood the original neighborhood to apply
   * @param _nhEval the evaluation function of the neighborhood
   */
  moEvaluatedNeighborhood(moNeighborhood<Neighbor> & _originalNeighborhood, moNeighborhoodEvaluation<Neighbor> & _nhEval) :
    moNeighborhood<Neighbor>(), originalNeighborhood(_originalNeighborhood), nhEval(_nhEval) {
  }
  
  /**
   * @return true if the neighborhood is random (default false)
   */
  virtual bool isRandom() {
    return originalNeighborhood.isRandom();
  }

  /**
   * Test if a neighbor exists
   * @param _solution the solution to explore
   * @return true if the neighborhood was not empty
   */
  virtual bool hasNeighbor(EOT& _solution) {
    return originalNeighborhood.hasNeighbor(_solution);
  }

  /**
   * Initialization of the neighborhood with the full evaluation
   *
   * @param _solution the solution to explore
   * @param _neighbor the first neighbor
   */
  virtual void init(EOT & _solution, Neighbor & _neighbor) {
    // full evaluation of the neighborhood
    nhEval(_solution);
    // initialisation of the original neighborhood
    originalNeighborhood.init(_solution, _neighbor);
  }

  /**
   * Give the next neighbor with the original neighborhood
   * @param _solution the solution to explore
   * @param _neighbor the next neighbor
   */
  virtual void next(EOT & _solution, Neighbor & _neighbor) {
    originalNeighborhood.next(_solution, _neighbor);
  }

  /**
   * give the continuation with the original neighborhood
   *
   * @param _solution the solution to explore
   * @return true if there is again a neighbor to explore
   */
  virtual bool cont(EOT & _solution) {
    return originalNeighborhood.cont(_solution);
  }

  /**
   * Return the class Name
   * @return the class name as a std::string
   */
  virtual std::string className() const {
    return "moEvaluatedNeighborhood";
  }

protected:
  moNeighborhood<Neighbor> & originalNeighborhood;
  moNeighborhoodEvaluation<Neighbor> & nhEval ;

};

#endif
