
#ifndef __moRealNeighborhood_h__
#define __moRealNeighborhood_h__

#include <mo>

template<class Distrib, class Neighbor>
class moRealNeighborhood : public moRndNeighborhood< Neighbor >, public eoFunctorBase
{
public:
    typedef typename Distrib::EOType EOT;

protected:
    Distrib & _distrib;
    edoSampler<Distrib> & _sampler;
    edoBounder<EOT> & _bounder;

public:

    moRealNeighborhood( Distrib & distrib, edoSampler<Distrib> & sampler, edoBounder<EOT> & bounder ) : _distrib(distrib), _sampler(sampler), _bounder(bounder) {}

    /**
     * It alway remains at least a solution in an infinite neighborhood
     * @param _solution the related solution
     * @return true
     */
    virtual bool hasNeighbor(EOT &)
    {
        return true;
    }

    /**
     * Draw the next neighbor
     * @param _solution the solution to explore
     * @param _current the next neighbor
     */
    virtual void next(EOT &, Neighbor & _current)
    {
        _current.bounder( &_bounder );

        // Draw a translation in the distrib, using the sampler
        _current.translation( _sampler( _distrib ) );
    }


    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _current)
    {
        // there is no difference between an init and a random draw
        next( _solution, _current );
    }
    /**
     * There is always a solution in an infinite neighborhood
     * @param _solution the solution to explore
     * @return true
     */
    virtual bool cont(EOT &)
    {
        return true;
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moRealNeighborhood";
    }

};

#endif // __moRealNeighborhood_h__
