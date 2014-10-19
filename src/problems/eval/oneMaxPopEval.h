/*
<oneMaxPopEval.h>
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

#ifndef _oneMaxPopEval_h
#define _oneMaxPopEval_h

//#include <problems/bitString/moPopSol.h> // ?
#include "oneMaxEval.h"
#include <cmath>

template< class EOT >
class oneMaxPopEval : public eoEvalFunc< moPopSol<EOT> >
{
public:

	oneMaxPopEval(oneMaxEval<EOT>& _eval, unsigned int _p): eval(_eval), p(_p){}
	/**
	 * Count the number of 1 in a bitString
	 * @param _sol the solution to evaluate
	 */
    void operator() (moPopSol<EOT>& _sol) {
    	double fit=0;
        for (unsigned int i = 0; i < _sol.size(); i++){
        	if(_sol[i].invalid())
        		eval(_sol[i]);
        	fit+=pow((double) _sol[i].fitness(), (int) p);
        }
        fit=pow((double) fit, (double)1/p);
        _sol.fitness(fit);
    }

private:
    oneMaxEval<EOT>& eval;
    unsigned int p;
};

#endif
