#ifndef _emptyNeighbor_h
#define _emptyNeighbor_h

#include <neighborhood/moNeighbor.h>

/*
  contener of the neighbor information
*/
template< class EOT , class Fitness >
class moEmptyNeighbor : public moNeighbor<EOT,Fitness>
{
public:
    typedef EOT EOType ;

    // empty constructor
    moEmptyNeighbor() : moNeighbor<EOType, Fitness>() { } ;

    /*
      make the evaluation of the current neighbor and update the information on this neighbor
    */
    virtual void eval(EOT & solution) { }

    /*
      move the solution
    */
    virtual void move(EOT & solution) { }

    // true if the this is better than the neighbor __neighbor
    virtual bool betterThan(moNeighbor<EOT,Fitness> & __neighbor) { return true; }
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
