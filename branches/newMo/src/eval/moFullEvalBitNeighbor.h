#ifndef _fullEvalBitNeighbor_h
#define _fullEvalBitNeighbor_h

#include <neighborhood/moBitNeighbor.h>
#include <ga.h>

/**
 * contener of the neighbor information
 */
template< class Fitness >
class moFullEvalBitNeighbor : public moBitNeighbor<Fitness>
{
public:
    typedef eoBit<Fitness> EOType ;

    using moBitNeighbor<Fitness>::bit ;

    // empty constructor needed
    moFullEvalBitNeighbor() : moBitNeighbor<Fitness>() { } ;

    moFullEvalBitNeighbor(unsigned b) : moBitNeighbor<Fitness>(bit) { } ;

    /*
      make the evaluation of the current neighbor and update the information on this neighbor
    */
    virtual void eval(EOType & solution) {
	Fitness fit = solution.fitness();

	solution[bit] = solution[bit]?false:true ;

	(*fullEval)(solution);

	fitness(solution.fitness());

	solution[bit] = solution[bit]?false:true ;

	solution.fitness(fit);
    };

    static void setFullEvalFunc(eoEvalFunc<EOType> & eval) {
	fullEval = & eval ;
    }

    static eoEvalFunc<EOType> * fullEval ;

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moFullEvalBitNeighbor"; }
};

template<class Fitness>
eoEvalFunc< eoBit<Fitness> > * moFullEvalBitNeighbor<Fitness>::fullEval = NULL ;

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
