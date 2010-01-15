#ifndef _fullEvalNeighbor_h
#define _fullEvalNeighbor_h

/*
  neighbor with full evaluation
*/
template< class EOT , class Fitness >
class moFullEvalNeighbor : moNeighbor<EOT, Fitness>
{
public:
    // empty constructor
    moFullEvalNeighbor(eoEvalFunc<EOT> & _eval) : fulleval(_eval) { } ;

    /*
      make the evaluation of the current neighbor and update the information on this neighbor
    */
    virtual void eval(EOT & solution) {
	Fitness fit = solution.fitness();

	move(solution);

	fulleval(solution);

	moveBack(solution);

	fitness = solution.fitness();

	solution.fitness(fit);
    };

    virtual moveBack(EOT & solution) ;

private:
    eoEvalFunc<EOT> & fulleval ;
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
