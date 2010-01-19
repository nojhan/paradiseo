#ifndef _moLocalSearch_h
#define _moLocalSearch_h

#include <explorer/moNeighborhoodExplorer.h>
#include <continuator/moContinuator.h>

/**
 * the main algorithm of the local search
 */
template< class NHE , class C >
class moLocalSearch: public eoMonOp<typename NHE::EOT>
{
public:
    typedef NHE NeighborhoodExplorer ;
    typedef C Continuator ;
    typedef typename NeighborhoodExplorer::EOT EOT ;
    

    /**
     * Constructor of a moLocalSearch needs a NeighborhooExplorer and a Continuator
     */
    moLocalSearch(NeighborhoodExplorer& _searchExpl, Continuator & _continuator) : searchExplorer(_searchExpl), continuator(_continuator) { } ;

    /**
     * Run the local search on a solution
     * @param _solution the related solution
     */
    virtual bool operator() (EOT & _solution) {
	
	// initialization of the external continuator (for example the time, or the number of generations)
	continuator.init(_solution);

	// initialization of the parameter of the search (for example fill empty the tabu list)
	searchExplorer.initParam(_solution);

	unsigned num = 0;

	do {
	    // explore the neighborhood of the solution
	    searchExplorer(_solution);

	    // if a solution in the neighborhood can be accepted
	    if (searchExplorer.accept(_solution))
		searchExplorer.move(_solution);

	    // update the parameter of the search (for ex. Temperature of the SA)
	    searchExplorer.updateParam(_solution);

	    std::cout << num << " : "  << _solution << std::endl ;
	    num++;
	} while (continuator(_solution) && searchExplorer.isContinue(_solution));

	searchExplorer.terminate(_solution);

	//A CHANGER
	return true;

    };

private:
    // make the exploration of the neighborhood according to a local search heuristic
    NeighborhoodExplorer& searchExplorer ;

    // external continuator
    Continuator& continuator ;
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
