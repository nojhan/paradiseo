#ifndef moFullEvalByCopy_H
#define moFullEvalByCopy_H

#include <eoEvalFunc.h>
#include <moEval.h>


/**
 * Evaluation by copy
 */
template<class Neighbor>
class moFullEvalByCopy : public moEval<Neighbor>
{
public:
     using moEval<Neighbor>::EOT EOT;
     using moEval<Neighbor>::Fitness Fitness;

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
    	 _neighbor.(tmp);
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
