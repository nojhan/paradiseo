#ifndef _moLocalSearch_h
#define _moLocalSearch_h

#include <explorer/moNeighborhoodExplorer.h>
#include <continuator/moContinuator.h>

/*
  the main algorithm of the local search
*/
template< class NHE , class C >
class moLocalSearch: public eoMonOp<typename NHE::EOT>
{
public:
    typedef NHE NeighborhoodExplorer ;
    typedef C Continuator ;
    typedef typename NeighborhoodExplorer::EOT EOT ;
    
    moLocalSearch(NeighborhoodExplorer & __searchExpl, Continuator & __continuator) : searchExplorer(__searchExpl), continuator(__continuator) { } ;

    virtual bool operator() (EOT & solution) {
	
	// initialization of the external continuator (for example the time, or the number of generations)
	continuator.init(solution);

	// initialization of the parameter of the search (for example fill empty the tabu list)
	searchExplorer.initParam(solution);

	unsigned num = 0;

	do {
	    // explore the neighborhood of the solution
	    searchExplorer(solution);

	    // if a solution in the neighborhood can be accepted
	    if (searchExplorer.accept(solution)) 
		searchExplorer.move(solution);

	    // update the parameter of the search (for ex. Temperature of the SA)
	    searchExplorer.updateParam(solution);

	    std::cout << num << " : "  << solution << std::endl ;
	    num++;
	} while (continuator(solution) && searchExplorer.isContinue(solution));

	searchExplorer.terminate(solution);

    };

private:
    // make the exploration of the neighborhood according to a local search heuristic
    NeighborhoodExplorer & searchExplorer ;

    // external continuator
    Continuator & continuator ;
};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
