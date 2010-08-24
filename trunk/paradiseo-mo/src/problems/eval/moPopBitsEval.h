/*
  <moPopBitsEval.h>
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

#ifndef moPopBitsEval_H
#define moPopBitsEval_H

#include <eoFunctor.h>
#include <eval/moEval.h>

/**
 * Class to compute the fitness of the solution-set after one bit flip of several solutions
 */
template<class Neighbor>
class moPopBitsEval : public moEval<Neighbor>
{
public:
  typedef typename Neighbor::EOT EOT;
  typedef typename Neighbor::SUBEOT SUBEOT; // type of the solution

  typedef typename SUBEOT::Fitness SUBFitness; // fitness type of the solution

  /**
   * Default constructor
   *
   * @param _eval evaluation function of the solution
   * @param _p exponent of the p-norm to compute the fitness of the solution-set 
   */
  moPopBitsEval(eoEvalFunc<SUBEOT>& _eval, unsigned int _p) : eval(_eval), p(_p) {
  }

  /**
   * Compute the fitness of the neighbor after the bit flip mutations
   *
   * @param _sol the solution-set
   * @param _n the neighbor which is supposed to be a "bits" neighbor
   */
  void operator()(EOT& _sol, Neighbor& _n) {
    double fit = 0;

    for(unsigned int i = 0; i < _sol.size(); i++) {
      if (_n.mutate[i]) {
	// save the fitness of the solution i
	SUBFitness f = _sol[i].fitness();
	
	// modify the solution
	_sol[i][ _n.bits[i] ] = !_sol[i][ _n.bits[i] ];
	_sol[i].invalidate();

	// evaluation of the solution
	eval(_sol[i]);
	
	// save the fitness in the neighbor
	_n.fitSol[i] = _sol[i].fitness();
	
	// compute the fitness of the solution-set
	fit += pow((double) _sol[i].fitness(), (int) p);
	
	// come back to the fitness of the initial solution
	_sol[i].fitness(f);	
	_sol[i][ _n.bits[i] ] = !_sol[i][ _n.bits[i] ];
      } else 
	// compute the fitness of the solution-set
	fit += pow((double) _sol[i].fitness(), (int) p);
	
    }

    fit = pow((double) fit, (double)1/p);
	
    _n.fitness(fit);
  }

private:
  eoEvalFunc<SUBEOT> & eval;
  unsigned int p;

};

#endif
