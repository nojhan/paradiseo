#ifndef _moNeighborComparator_h
#define _moNeighborComparator_h

#include <neighborhood/moNeighbor.h>

template< class Neigh >
class moNeighborComparator : public eoBF<const Neigh & , const Neigh & , bool>
{
public:

    /*
    * true if the neighbor1 is better than neighbor2
    */
    virtual bool operator()(const Neigh & neighbor1, const Neigh & neighbor2) {
	return (neighbor1.fitness() > neighbor2.fitness()); 
    }

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moNeighborComparator"; }
};


#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
