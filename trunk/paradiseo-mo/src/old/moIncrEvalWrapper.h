#ifndef moIncrEvalWrapper_H
#define moIncrEvalWrapper_H

#include <eoEvalFunc.h>
#include <moEval.h>

/*
 * (Old fashioned) Incremental evaluation to use with a moMoveNeighbor
 * WARNING: Don't use this class unless you are an moMove user.
 */
template<class MoveNeighbor, class M>
class moIncrEvalWrapper : public moEval<MoveNeighbor>
{
public:
    using moEval<BackableNeighbor>::EOT EOT;
    using moEval<BackableNeighbor>::Fitness Fitness;

    moIncrEvalWrapper(moIncrEval<M>& _incr):incr(_incr) {}

    /*
    * make the evaluation of the current neighbor and update the information on this neighbor
    * the evaluation could be incremental
    */
    virtual void eval(MoveNeighbor& _neighbor,EOT & _solution) {
        _neighbor.fitness(incrEval(*(_neighbor.getMove()), _solution));
    }

private:
    /** the full evaluation object */
    moIncrEval<M> & incrEval;

};

#endif
