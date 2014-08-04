#ifndef moeoFullEvalByCopy_H
#define moeoFullEvalByCopy_H

#include "../../eo/eoEvalFunc.h"
#include "../../mo/eval/moEval.h"

/**
 * Evaluation by copy
 */
template<class Neighbor>
class moeoFullEvalByCopy : public moEval<Neighbor>
{
public:
    typedef typename moEval<Neighbor>::EOT EOT;
    typedef typename moEval<Neighbor>::Fitness Fitness;

    /**
     * Ctor
     * @param _eval the full evaluation object
     */
    moeoFullEvalByCopy(eoEvalFunc<EOT> & _eval) : eval(_eval) {}

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
        _neighbor.fitness(tmp.objectiveVector());
    }


private:
    /** the full evaluation object */
    eoEvalFunc<EOT> & eval;

};

#endif
