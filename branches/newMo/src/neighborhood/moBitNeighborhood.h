#ifndef _bitNeighborhood_h
#define _bitNeighborhood_h

#include <neighborhood/moNeighborhood.h>

template< class N >
class moBitNeighborhood : public moNeighborhood<N>
{
public:
    typedef N Neighbor ;
    typedef typename Neighbor::EOType EOT ;

    moBitNeighborhood() : moNeighborhood<Neighbor>() { }

    virtual bool hasNeighbor(EOT & solution) {

      return true;

    }

    /*
      initialisation of the neighborhood
    */
    virtual void init(EOT & solution, Neighbor & _neighbor) {
	currentBit = 0 ;

	_neighbor.bit = currentBit ;
    } 

    /*
    Give the next neighbor
    */
    virtual void next(EOT & solution, Neighbor & neighbor) { 
	currentBit++ ;

	neighbor.bit = currentBit ;	
    } 

    /*
    if false, there is no neighbor left to explore
    */
    virtual bool cont(EOT & solution) { 
	return (currentBit < solution.size()) ;
    } 
    
private:
    unsigned currentBit;
};


#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
