#ifndef _oneMaxBitNeighbor_h
#define _oneMaxBitNeighbor_h

#include <neighborhood/moBitNeighbor.h>
#include <ga.h>

/*
  contener of the neighbor information
*/
template< class Fitness >
class OneMaxBitNeighbor : public moBitNeighbor<Fitness>
{
public:
    typedef eoBit<Fitness> EOType ;

    using moBitNeighbor<Fitness>::bit ;

    /*
    * incremental evaluation of the solution for the oneMax problem
    */
    virtual void eval(EOType & solution) {
        if (solution[bit] == 0)
            fitness(solution.fitness() + 1);
        else
            fitness(solution.fitness() - 1);
    };
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
