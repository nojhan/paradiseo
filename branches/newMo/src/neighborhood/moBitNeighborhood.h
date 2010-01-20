#ifndef _bitNeighborhood_h
#define _bitNeighborhood_h

#include <neighborhood/moNeighborhood.h>

/**
 * Neighborhood related to a vector of Bit
 */
template< class N >
class moBitNeighborhood : public moNeighborhood<N>
{
public:
    typedef N Neighbor ;
    typedef typename Neighbor::EOT EOT ;

    /**
     * Default Constructor
     */
    moBitNeighborhood() : moNeighborhood<Neighbor>() { }

    /**
     * Test if it exist a neighbor
     * @param _solution the solution to explore
     * @return always True
     */
    virtual bool hasNeighbor(EOT& _solution) {
    	return true;
    }

    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _neighbor the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _neighbor) {
		currentBit = 0 ;
		_neighbor.bit = currentBit ;
    } 

    /**
     * Give the next neighbor
     * @param _solution the solution to explore
     * @param _neighbor the next neighbor
     */
    virtual void next(EOT & _solution, Neighbor & _neighbor) {
		currentBit++ ;
		_neighbor.bit = currentBit ;
    } 

    /**
     * test if all neighbors are explore or not,if false, there is no neighbor left to explore
     * @param _solution the solution to explore
     * @return true if there is again a neighbor to explore
     */
    virtual bool cont(EOT & _solution) {
    	return (currentBit < _solution.size()) ;
    } 
    
private:
    //Position in the neighborhood
    unsigned int currentBit;
};


#endif

