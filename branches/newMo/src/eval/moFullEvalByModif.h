#ifndef moFullEvalByModif_H
#define moFullEvalByModif_H

#include <eoEvalFunc.h>
#include <moEval.h>

/**
 * Full evaluation to use with a moBackableNeighbor
 */
template<class BackableNeighbor>
class moFullEvalByModif : public moEval<BackableNeighbor>
{
public:
     using moEval<BackableNeighbor>::EOT EOT;
     using moEval<BackableNeighbor>::Fitness Fitness;

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
     void operator()(EOT & _sol, BackableNeighbor & _neighbor)
     {
    	 // tmp fitness value of the current solution
    	 Fitness tmpFit;

    	 // save current fitness value
    	 tmpFit = _sol.fitness();

    	 // move the current solution wrt _neighbor
    	 _neighbor.move(_sol);
    	 // eval the modified solution
    	 _sol.invalidate();
    	 eval(_sol);
    	 // set the fitness value to the neighbor
    	 _neighbor.fitness(_sol.fitness());
    	 // move the current solution back
    	_neighbor.moveBack(_sol);
    	// set the fitness back
    	_sol.fitness(tmpFit);
     }


private:
     /** the full evaluation object */
     eoEvalFunc<EOT> & eval;

};

#endif
