/*
<moMonOpPerturb.h>
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

#ifndef _moMonOpPerturb_h
#define _moMonOpPerturb_h

#include <eoEvalFunc.h>
#include <eoOp.h>
#include <perturb/moPerturbation.h>
#include <memory/moDummyMemory.h>

/**
 * Perturbation operator using only a eoMonOp
 */
template< class Neighbor >
class moMonOpPerturb : public moPerturbation<Neighbor>, public moDummyMemory<Neighbor> {

public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _monOp an eoMonOp (pertubation operator)
     * @param _fullEval a full evaluation function
     * @param _nbPerturbation number of operator executions for perturbation
     */
  moMonOpPerturb(eoMonOp<EOT>& _monOp, eoEvalFunc<EOT>& _fullEval, unsigned int _nbPerturbation = 1):monOp(_monOp), fullEval(_fullEval), nbPerturbation(_nbPerturbation) {}

    /**
     * Apply monOp on the solution
     * @param _solution to perturb
     * @return value of monOp
     */
    bool operator()(EOT& _solution) {
      bool res = false;

      for(unsigned int i = 0; i < nbPerturbation; i++) 
	res = monOp(_solution) || res;
      
      _solution.invalidate();
      fullEval(_solution);

      return res;
    }

private:
  /** monOp */
  eoMonOp<EOT>& monOp;
  eoEvalFunc<EOT>& fullEval;
  unsigned int nbPerturbation;
};

#endif
