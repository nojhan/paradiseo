/*
  <moUBQPdoubleIncrEvaluation.h>
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

#ifndef moUBQPdoubleIncrEvaluation_H
#define moUBQPdoubleIncrEvaluation_H

#include "moDoubleIncrEvaluation.h"
#include "../../explorer/moNeighborhoodExplorer.h"
#include "../../eval/moEval.h"

/**
 * The neighborhood evaluation for the UBQP
 * The double incremental evaluation is used
 * 
 * BECAREFULL: This object must be added to the moCheckpoint of the local search (init method)
 */
template<class Neighbor>
class moUBQPdoubleIncrEvaluation : public moDoubleIncrEvaluation<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT;
  typedef typename EOT::Fitness Fitness;

  using moDoubleIncrEvaluation<Neighbor>::deltaFitness;
  using moDoubleIncrEvaluation<Neighbor>::firstEval;

  /**
   * Constructor 
   *
   * @param _neighborhoodSize the size of the neighborhood
   * @param _incrEval the incremental evaluation of the UBQP
   */
  moUBQPdoubleIncrEvaluation(unsigned int _neighborhoodSize, moUBQPSimpleIncrEval<Neighbor> & _incrEval) : moDoubleIncrEvaluation<Neighbor>(_neighborhoodSize), searchExplorer(NULL)
  {
    n = _incrEval.getNbVar();
    Q = _incrEval.getQ();
  }
  
  void neighborhoodExplorer(moNeighborhoodExplorer<Neighbor> & _searchExplorer) {
    searchExplorer = & _searchExplorer;
  }

  /**
   *  Evaluation of the neighborhood
   *  Here nothing to do 
   *
   * @param _solution the current solution 
   */
  virtual void operator()(EOT & _solution) {
    if (firstEval) {
      firstEval = false;

      // compute the delta in the simple incremental way O(n)
      unsigned int j;
      int d;
      for(unsigned i = 0; i < n; i++) {
	d = Q[i][i]; 

	for(j = 0; j < i; j++)
	  if (_solution[j])
	    d += Q[i][j];
	
	for(j = i+1; j < n; j++)
	  if (_solution[j])
	    d += Q[j][i];
	
	if (_solution[i])
	  d = - d;
	
	deltaFitness[i] = d;
      }
    } else {

      if (searchExplorer->moveApplied()) { 
	// compute the new fitness only when the solution has moved
	// the selectedNeighbor is the neighbor which is selected in the neighborhood
	// the movement is made on this neighbor
	// we suppose that the neighborhood is bit string neighborhood (indexed neighbor)
	unsigned iMove = searchExplorer->getSelectedNeighbor().index();

	for(unsigned i = 0; i < n; i++) {
	  if (i == iMove)
	    deltaFitness[i] = - deltaFitness[i] ;
	  else {
	    if (_solution[i] != _solution[iMove])
	      if (i < iMove) 
		deltaFitness[i] += Q[iMove][i];
	      else
		deltaFitness[i] += Q[i][iMove];
	    else
	      if (i < iMove) 
		deltaFitness[i] -= Q[iMove][i];
	      else
		deltaFitness[i] -= Q[i][iMove];
	  }
	}
      }     
    }
  }

  /*
   * to get the matrix Q
   *
   * @return matrix Q
   */
  int** getQ() {
    return Q;
  }

  /*
   * to get the number of variable (bit string length)
   *
   * @return bit string length
   */
  int getNbVar() {
    return n;
  }

private:
  // number of variables
  int n;
  
  // matrix Q (supposed to be in lower triangular form)
  int ** Q;

  /** The search explorer of the local search */
  moNeighborhoodExplorer<Neighbor> * searchExplorer;
};

#endif
