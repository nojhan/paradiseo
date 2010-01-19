#ifndef _moNeighborComparator_h
#define _moNeighborComparator_h

#include <EO.h>
#include <eoFunctor.h>

#include <neighborhood/moNeighbor.h>

template< class Neighbor >
class moNeighborComparator : public eoBF<const Neighbor & , const Neighbor & , bool>
{
public:

    /*
     * Compare two neighbors
     * @param _neighbor1 the first neighbor
     * @param _neighbor2 the second neighbor
     * @return true if the neighbor1 is better than neighbor2
     */
    virtual bool operator()(const Neighbor& _neighbor1, const Neighbor& _neighbor2) {
    	return (neighbor1.fitness() > neighbor2.fitness());
    }

    /*
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
    	return "moNeighborComparator";
    }
};


#endif
