/*
<moSA.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau, Lionel Parreaux

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

#ifndef _moSA_h
#define _moSA_h

#include <algo/moLocalSearch.h>
#include <explorer/moSAExplorer.h>
#include <coolingSchedule/moCoolingSchedule.h>
#include <coolingSchedule/moSimpleCoolingSchedule.h>
#include <continuator/moTrueContinuator.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>
#include <eoOptional.h>

/**
 * Simulated Annealing
 */
template<class Neighbor>
class moSA: public moLocalSearch<Neighbor>
{
public:

    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;


    /**
     * Basic constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _initT initial temperature for cooling schedule (default = 10)
     * @param _alpha factor of decreasing for cooling schedule (default = 0.9)
     * @param _span number of iteration with equal temperature for cooling schedule (default = 100)
     * @param _finalT final temperature, threshold of the stopping criteria for cooling schedule (default = 0.01)
     */
    moSA(
        Neighborhood& _neighborhood,
        eoEvalFunc<EOT>& _fullEval,
        moEval<Neighbor>& _eval,
        double _initT=10,
        double _alpha=0.9,
        unsigned _span=100,
        double _finalT=0.01
    )
    : moLocalSearch<Neighbor> (
        *(explorer_ptr = defaultExplorer = new moSAExplorer<Neighbor>(_neighborhood, _eval, *(defaultCool = new moSimpleCoolingSchedule<EOT>(_initT, _alpha, _span, _finalT)), NULL)),
        *(defaultContinuator = new moTrueContinuator<Neighbor>()),
        _fullEval ),
      explorer(*explorer_ptr),
      defaultEval(NULL)             // removed in C++11 with unique_ptr
    { }
    
    /**
     * General constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _cool a cooling schedule
     * @param _eval neighbor's evaluation function
     * @param _cont an external continuator
     * @param _comp a solution vs neighbor comparator
     */
    moSA (
        Neighborhood& _neighborhood,
        eoEvalFunc<EOT>& _fullEval,
        moCoolingSchedule<EOT>& _cool,
        eoOptional< moEval<Neighbor> > _eval                  = NULL,
        eoOptional< moContinuator<Neighbor> > _cont           = NULL,
        eoOptional< moSolNeighborComparator<Neighbor> > _comp = NULL
    )
    : moLocalSearch<Neighbor>  (
        *(explorer_ptr = defaultExplorer = new moSAExplorer<Neighbor> (
            _neighborhood,
            _eval.hasValue()? defaultEval = NULL, _eval.get(): *(defaultEval = new moFullEvalByCopy<Neighbor>(_fullEval)),
            // C++11: _eval.hasValue()? _eval.get(): *(defaultEval = new moFullEvalByCopy<Neighbor>(_fullEval)),
            _cool,
            _comp )),
                        // ),
        _cont.hasValue()? defaultContinuator = NULL, _cont.get(): *(defaultContinuator = new moTrueContinuator<Neighbor>()),
        _fullEval  ),
      explorer(*explorer_ptr),
      defaultCool(NULL)              // removed in C++11 with unique_ptr
    { }
    
    /**
     * More general constructor for a simulated annealing, giving the explorer explicitly
     * @param _fullEval the full evaluation function
     * @param _explorer the SA explorer
     * @param _cont an external continuator
     */
    moSA (
        eoEvalFunc<EOT>& _fullEval,
        moSAExplorer<Neighbor>& _explorer,
        eoOptional< moContinuator<Neighbor> > _cont = NULL
    )
    : moLocalSearch<Neighbor>  (
        *(explorer_ptr = &_explorer),
        _cont.hasValue()? defaultContinuator = NULL, _cont.get(): *(defaultContinuator = new moTrueContinuator<Neighbor>()), _fullEval  ),
      defaultEval(NULL),              // removed in C++11 with unique_ptr
      defaultExplorer(NULL),          // removed in C++11 with unique_ptr
      explorer(*explorer_ptr),
      defaultCool(NULL)               // removed in C++11 with unique_ptr
    { }
    
    virtual ~moSA ()
    {
        // Note: using unique_ptr would allow us to remove this explicit destructor, but they were only introduced in C++11
        if (defaultContinuator != NULL)
            delete defaultContinuator;
        if (defaultCool != NULL)
            delete defaultCool;
        if (defaultExplorer != NULL)
            delete defaultExplorer;
        if (defaultEval != NULL)
            delete defaultEval;
    }

private:
    moFullEvalByCopy<Neighbor>* defaultEval;
    moSAExplorer<Neighbor>* defaultExplorer;
    moSAExplorer<Neighbor>* explorer_ptr;             // Should never be NULL
    moSAExplorer<Neighbor>& explorer;                 // Not very useful field (not used yet)
    moSimpleCoolingSchedule<EOT>* defaultCool;        // C++11: const std::unique_ptr<moSimpleCoolingSchedule<EOT>> defaultCool;
    moTrueContinuator<Neighbor>* defaultContinuator;
};

#endif


