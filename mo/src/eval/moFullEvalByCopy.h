	/*
  <moFullEvalByCopy.h>
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

#ifndef moFullEvalByCopy_H
#define moFullEvalByCopy_H

#include <paradiseo/eo/eoEvalFunc.h>
#include "moEval.h"

/**
 * Evaluation by copy
 */
template<class Neighbor>
class moFullEvalByCopy : public moEval<Neighbor>
{
public:
    typedef typename moEval<Neighbor>::EOT EOT;
    typedef typename moEval<Neighbor>::Fitness Fitness;

    /**
     * Ctor
     * @param _eval the full evaluation object
     */
    moFullEvalByCopy(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

    /**
     * Full evaluation of the neighbor by copy
     * @param _sol current solution
     * @param _neighbor the neighbor to be evaluated
     */
    void operator()(EOT & _sol, Neighbor & _neighbor)
    {
        // tmp solution
        EOT tmp(_sol);
        // move tmp solution wrt _neighbor
        _neighbor.move(tmp);
        // eval copy
        tmp.invalidate();
        eval(tmp);
        // set the fitness value to the neighbor
        _neighbor.fitness(tmp.fitness());
    }


private:
    /** the full evaluation object */
    eoEvalFunc<EOT> & eval;

};

#endif
