#ifndef _neighborhoodExplorer_h
#define _neighborhoodExplorer_h

//EO inclusion
#include <eoFunctor.h>

#include <neighborhood/moNeighborhood.h>

/**
 * Explore the neighborhood
 */
template< class NH >
class moNeighborhoodExplorer : public eoUF<typename NH::EOT&, void>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;
    typedef typename Neighborhood::Neighbor Neighbor ;

    /**
     * Constructor with a Neighborhood and evaluation function
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     */
    moNeighborhoodExplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval):neighborhood(_neighborhood), eval(_eval) {}

    /**
     * Init Search parameters
     * @param _solution the solution to explore
     */
    virtual void initParam (EOT& _solution) = 0 ;

    /**
     * Update Search parameters
     * @param _solution the solution to explore
     */
    virtual void updateParam (EOT& _solution) = 0 ;

    /**
     * Test if the exploration continue or not
     * @param _solution the solution to explore
     * @return true if the exploration continue, else return false
     */
    virtual bool isContinue(EOT& _solution) = 0 ;

    /**
     * Move a solution
     * @param _solution the solution to explore
     */
    virtual void move(EOT& _solution) = 0 ;

    /**
     * Test if a solution is accepted
     * @param _solution the solution to explore
     * @return true if the solution is accepted, else return false
     */
    virtual bool accept(EOT& _solution) = 0 ;

    /**
     * Terminate the search
     * @param _solution the solution to explore
     */
    virtual void terminate(EOT& _solution) = 0 ;

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
    	return "moNeighborhoodExplorer";
    }

protected:
    Neighborhood & neighborhood;
    moEval<Neighbor>& eval;

};

#endif
