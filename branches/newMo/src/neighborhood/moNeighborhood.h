#ifndef _moNeighborhood_h
#define _moNeighborhood_h

template< class Neigh >
class moNeighborhood : public eoObject
{
public:
    typedef Neigh Neighbor;
    typedef typename Neighbor::EOType EOT;

    moNeighborhood() { }

    virtual bool hasNeighbor(EOT & solution) = 0 ;

    /*
      initialisation of the neighborhood
    */
    virtual void init(EOT & solution, Neighbor & current) = 0 ;

    /*
    Give the next neighbor
    */
    virtual void next(EOT & solution, Neighbor & current) = 0 ; 

    /*
    if false, there is no neighbor left to explore
    */
    virtual bool cont(EOT & solution) = 0 ;

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moNeighborhood"; }
};

#endif
