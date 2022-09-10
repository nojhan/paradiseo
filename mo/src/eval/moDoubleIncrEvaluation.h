/*
  <moDoubleIncrEvaluation.h>
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

#ifndef moDoubleIncrEvaluation_H
#define moDoubleIncrEvaluation_H

#include <eval/moNeighborhoodEvaluation.h>
#include <continuator/moUpdater.h>

/**
 * Base class for the double incremental evaluation of the neighborhood
 *
 * The sizes of the neighborhoods are equal
 */
template<class Neighbor>
class moDoubleIncrEvaluation : public moNeighborhoodEvaluation<Neighbor>, public moUpdater
{
public:
  typedef typename Neighbor::EOT EOT;
  typedef typename EOT::Fitness Fitness;

  /**
   * Constructor 
   *
   * @param _neighborhoodSize the size of the neighborhood
   */
  moDoubleIncrEvaluation(unsigned int _neighborhoodSize) : moNeighborhoodEvaluation<Neighbor>(), moUpdater(), neighborhoodSize(_neighborhoodSize), firstEval(true) { 
    deltaFitness = new Fitness[neighborhoodSize];
  }
  
  /**
   * Destructor
   */
  ~moDoubleIncrEvaluation() {
    if (deltaFitness != NULL)
      delete [] deltaFitness;
  }

  /**
   * Initialisation of the evaluation process
   * The first evaluation will be a simple incremental evaluation
   *
   */
  virtual void init() {
    firstEval = true;
  }

  virtual void operator()() {
  }

  /**
   *  Evaluation of the neighborhood
   *  Here nothing to do 
   *
   * @param _solution the current solution 
   */
  virtual void operator()(EOT & /*_solution*/) {
  }

  /** the delta of fitness for each neighbors 
   *  The fitness of the neighbor i is given by : fitness(solution) + deltaFitness[i]
   */
  Fitness * deltaFitness;

protected:
  /** the neighborhood size */
  unsigned int neighborhoodSize;

  /** flag : true when it is the first evaluation */
  bool firstEval;
};

#endif
