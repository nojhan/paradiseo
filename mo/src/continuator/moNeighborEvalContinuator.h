/*
<moNeighborEvalContinuator.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

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

#ifndef _moNeighborEvalContinuator_h
#define _moNeighborEvalContinuator_h

#include "moContinuator.h"
#include "../neighborhood/moNeighborhood.h"
#include "../eval/moEvalCounter.h"

/**
 * Continue until a maximum fixed number of neighbor evaluation is reached
 *
 *
 * Becareful 1: if restartCounter is true, then the number of neighbor evaluations (for example incremental evaluations) is considered during the local search (not before it)
 *
 * Becareful 2: Can not be used if the evaluation function is used in parallel
 */
template< class Neighbor >
class moNeighborEvalContinuator : public moContinuator<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    /**
     * Constructor
     * @param _eval neighbor evaluation function to count
     * @param _maxNeighborEval number maximum of iterations
     * @param _restartCounter if true the counter of number of evaluations restarts to "zero" at initialization, if false, the number is cumulative
     */
  moNeighborEvalContinuator(moEvalCounter<Neighbor> & _eval, unsigned int _maxNeighborEval, bool _restartCounter = true): eval(_eval), maxNeighborEval(_maxNeighborEval), restartCounter(_restartCounter) {}

    /**
     * Test if continue
     * @param _solution a solution
     * @return true if number of evaluations < maxNeighborEval
     */
    virtual bool operator()(EOT & _solution) {
        return (eval.value() - nbEval_start < maxNeighborEval);
    }

    /**
     * Reset the number of evaluations
     * @param _solution a solution
     */
    virtual void init(EOT & _solution) {
      if (restartCounter)
        nbEval_start = eval.value();
      else
	nbEval_start = 0;
    }

    /**
     * the current number of evaluation from the begining
     * @return the number of evaluation
     */
    unsigned int value() {
        return eval.value() - nbEval_start ;
    }

private:
  moEvalCounter<Neighbor> & eval;
  unsigned int maxNeighborEval;
  bool restartCounter;
  unsigned int nbEval_start ;

};
#endif
