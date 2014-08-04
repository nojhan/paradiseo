/*
<moPopSolEval.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

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

#ifndef _moPopSolEval_h
#define _moPopSolEval_h

//#include <problems/bitString/moPopSol.h> // ?
#include "../../eo/eoEvalFunc.h"
#include <cmath>

/**
 * To compute the fitness of a pop-based solution
 */
template< class EOT >
class moPopSolEval : public eoEvalFunc<EOT>
{
public:
  typedef typename EOT::SUBEOT SUBEOT;

  /**
   * default constructor
   *
   * @param _eval evaluation function of the solution
   * @param _p the exponent of the p-norm to compute the population based fitness
   */
  moPopSolEval(eoEvalFunc<SUBEOT>& _eval, unsigned int _p): eval(_eval), p(_p){}
  
  /**
   * to compute the fitness of a population-based solution
   * which is the norm p of the fitness of solutions: ( sigma_{i} f(s_i)^p )^(1/p) 
   *
   * re-compute the fitness of solution which are invalid
   *
   * @param _sol the population-based solution to evaluate
   */
  void operator() (EOT& _sol) {
    double fit = 0;

    for (unsigned int i = 0; i < _sol.size(); i++) {
      if(_sol[i].invalid())
	eval(_sol[i]);

      fit += pow((double) _sol[i].fitness(), (int) p);
    }

    fit = pow((double) fit, (double)1/p);

    _sol.fitness(fit);
  }

private:
  eoEvalFunc<SUBEOT>& eval;
  unsigned int p;
};

#endif
