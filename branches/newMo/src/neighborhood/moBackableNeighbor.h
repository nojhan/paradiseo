#ifndef _BackableNeighbor_h
#define _BackableNeighbor_h

/**
 * Neighbor with a move back function to use in a moFullEvalByModif
 */
template< class EOT , class Fitness >
class moBackableNeighbor : public moNeighbor<EOT, Fitness>
{
public:

	/**
	 * the move back function
	 * @param _solution the solution to moveBack
	 */
    virtual void moveBack(EOT & _solution){}

};

#endif
