/*
<moMaxSATincrEval.h>
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

#ifndef _moMaxSATincrEval_h
#define _moMaxSATincrEval_h

#include "../../eval/moEval.h"
#include "../../../problems/eval/maxSATeval.h"

/**
 * Incremental evaluation Function for the max SAT problem
 */
template <class Neighbor>
class moMaxSATincrEval : public moEval <Neighbor> {
public :
    typedef typename Neighbor::EOT EOT;

    moMaxSATincrEval(MaxSATeval<EOT> & _fulleval) : fulleval(_fulleval) {
        nbClauses = _fulleval.nbClauses;
        nbVar     = _fulleval.nbVar;

        clauses   =  _fulleval.clauses;
        variables = _fulleval.variables;
    }

    /**
     * incremental evaluation of the neighbor for the max SAT problem
     * @param _solution the solution (of type bit string) to move
     * @param _neighbor the neighbor (of type moBitNeigbor) to consider
     */
    virtual void operator()(EOT & _solution, Neighbor & _neighbor) {
        // the difference of fitness
        int delta = 0;

        // the flipped bit
        unsigned int bit = _neighbor.index();

        // clauses which can be modified by the flipped bit
        const std::vector<int> & modifiedClauses = variables[bit + 1] ; // remember that the variables start at index 1 and not 0
        unsigned int size = modifiedClauses.size();

        int nc;
        bool litt;

        for (unsigned int k = 0; k < size; k++) {
            // number of the clause
            nc = modifiedClauses[k];

            // negative means that the not(variable) is in the clause
            if (nc < 0) {
                nc = - nc;
                litt = !_solution[bit];
            } else
                litt = _solution[bit];

            if (litt) {
                // the litteral was true and becomes false
                _solution[bit] = !_solution[bit];

                if (!fulleval.clauseEval(nc, _solution))
                    // the clause was true and becomes false
                    delta--;

                _solution[bit] = !_solution[bit];
            } else {
                // the litteral was false and becomes true
                if (!fulleval.clauseEval(nc, _solution))
                    // the clause was false and becomes true
                    delta++;
            }
        }

        _neighbor.fitness(_solution.fitness() + delta);
    }

protected:
    // number of variables
    unsigned int nbVar;
    // number of clauses
    unsigned int nbClauses;

    // list of clauses:
    //   each clause has the number of the variable (from 1 to nbVar)
    //   when the value, litteral = not(variable)
    std::vector<int> * clauses;

    // list of variables:
    //   for each variable, the list of clauses
    std::vector<int> * variables;

    //full eval of the max SAT
    MaxSATeval<EOT> & fulleval;
};

#endif
