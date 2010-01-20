#ifndef _moNeighborhood_h
#define _moNeighborhood_h

#include <eoObject.h>

/**
 * A Neighborhood
 */
template< class Neighbor >
class moNeighborhood : public eoObject
{
public:
	/**
	 * Define type of a solution corresponding to Neighbor
	 */
    typedef typename Neighbor::EOT EOT;

    /**
     * Default Constructor
     */
    moNeighborhood(){}

    /**
     * Test if a solution has (again) a Neighbor
     * @param _solution the related solution
     * @return if _solution has a Neighbor
     */
    virtual bool hasNeighbor(EOT & _solution) = 0 ;

    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _current) = 0 ;

    /**
     * Give the next neighbor
     * @param _solution the solution to explore
     * @param _current the next neighbor
     */
    virtual void next(EOT & _solution, Neighbor & _current) = 0 ;

    /**
     * Test if there is again a neighbor
     * @param _solution the solution to explore
     * @return if there is again a neighbor not explored
     */
    virtual bool cont(EOT & _solution) = 0 ;

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const { return "moNeighborhood"; }
};

#endif
