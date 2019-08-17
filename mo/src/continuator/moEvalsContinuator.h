/*
<moEvalsContinuator.h>
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

#ifndef _moEvalsContinuator_h
#define _moEvalsContinuator_h

#include "moContinuator.h"
#include "../neighborhood/moNeighborhood.h"
#include <paradiseo/eo/eoEvalFuncCounter.h>
#include "../eval/moEvalCounter.h"

/**
 * Continue until a maximum fixed number of full evaluation and neighbor evaluation is reached (total number of evaluation = full evaluation + incremental evaluation)
 *
 *
 * Becareful 1: if restartCounter is true, then the number of neighbor evaluations (for example incremental evaluations) is considered during the local search (not before it)
 *
 * Becareful 2: Can not be used if the evaluation function is used in parallel
 *
 * Becareful 3: Check if the incremental does not use full evaluation, otherwise the total number of evaluations is not correct
 */
template< class Neighbor >
class moEvalsContinuator : public moContinuator<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    /**
     * Constructor
     * @param _fullEval full evaluation function to count
     * @param _neighborEval neighbor evaluation function to count
     * @param _maxEvals number maximum of evaluations (full and incremental evaluations)
     * @param _restartCounter if true the counter of number of evaluations restarts to "zero" at initialization, if false, the number is cumulative
     */
  moEvalsContinuator(eoEvalFuncCounter<EOT> & _fullEval, moEvalCounter<Neighbor> & _neighborEval, unsigned int _maxEvals, bool _restartCounter = true): fullEval(_fullEval), neighborEval(_neighborEval), maxEvals(_maxEvals), restartCounter(_restartCounter) {}

    /**
     * Test if continue
     * @param _solution a solution
     * @return true if number of evaluations < maxEvals
     */
    virtual bool operator()(EOT & _solution) {
        return (fullEval.value() + neighborEval.value() - nbEval_start < maxEvals);
    }

    /**
     * Reset the number of evaluations
     * @param _solution a solution
     */
    virtual void init(EOT & _solution) {
      if (restartCounter)
        nbEval_start = fullEval.value() + neighborEval.value();
      else
	nbEval_start = 0;
    }

    /**
     * the current number of evaluation from the begining
     * @return the number of evaluation
     */
    unsigned int value() {
        return fullEval.value() + neighborEval.value() - nbEval_start ;
    }

private:
  eoEvalFuncCounter<EOT> & fullEval;
  moEvalCounter<Neighbor> & neighborEval;
  unsigned int maxEvals;
  bool restartCounter;
  unsigned int nbEval_start ;

};
#endif
