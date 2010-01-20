#ifndef _moSimpleHCexplorer_h
#define _moSimpleHCexplorer_h

#include <explorer/moNeighborhoodExplorer.h>

/**
 * Explorer for a simple Hill-climbing
 */
template< class Neighborhood >
class moSimpleHCexplorer : public moNeighborhoodExplorer<Neighborhood>
{
public:
    typedef typename Neighborhood::EOT EOT ;
    typedef typename Neighborhood::Neighbor Neighbor ;

    using moNeighborhoodExplorer<Neighborhood>::neighborhood;
    using moNeighborhoodExplorer<Neighborhood>::eval;
    using moNeighborhoodExplorer<Neighborhood>::comparator;

	/**
	 * Constructor
	 * @param _neighborhood the neighborhood
	 * @param _eval the evaluation function
	 * @param _comparator a neighbor comparator
	 */
    moSimpleHCexplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _comparator) : moNeighborhoodExplorer<Neighborhood>(_neighborhood, _eval, _comparator){
    	isAccept = false;
    	current=new Neighbor();
    	best=new Neighbor();
    }

	/**
	 * initParam: NOTHING TO DO
	 */
    virtual void initParam(EOT & solution){};

	/**
	 * updateParam: NOTHING TO DO
	 */
    virtual void updateParam(EOT & solution){};

	/**
	 * terminate: NOTHING TO DO
	 */
    virtual void terminate(EOT & solution){};

    /**
     * Explore the neighborhood of a solution
     * @param _solution
     */
    virtual void operator()(EOT & _solution){

	//est qu'on peut initializer
    	//Test if _solution has a Neighbor
		if(neighborhood.hasNeighbor(_solution)){
			//init the first neighbor
			neighborhood.init(_solution, (*current));

			//eval the _solution moved with the neighbor and stock the result in the neighbor
			eval(_solution, (*current));

			//initialize the best neighbor
			(*best) = (*current);

			//test all others neighbors
			while (neighborhood.cont(_solution)) {
				//next neighbor
				neighborhood.next(_solution, (*current));
				//eval
				eval(_solution, (*current));
				//if we found a better neighbor, update the best
				if (comparator((*current), (*best))) {
					(*best) = (*current);
				}
			}
		}
		else{
			//if _solution hasn't neighbor,
			isAccept=false;
		}
    };

    /**
     * continue if a move is accepted
     * @param _solution the solution
     * @return true if an ameliorated neighbor was be found
     */
    virtual bool isContinue(EOT & _solution) {
    	return isAccept ;
    };

    /**
     * move the solution with the best neighbor
     * @param _solution the solution to move
     */
    virtual void move(EOT & _solution) {
		//move the solution
    	(*best).move(_solution);
    	//update its fitness
    	_solution.fitness((*best).fitness());
    };

    /**
     * accept test if an amelirated neighbor was be found
     * @param _solution the solution
     * @return true if the best neighbor ameliorate the fitness
     */
    virtual bool accept(EOT & _solution) {
		if(neighborhood.hasNeighbor(_solution)){
			isAccept = (_solution.fitness() < (*best).fitness()) ;
		}
		return isAccept;
    };

private:

    // attention il faut que le constructeur vide existe
    //(Pointeurs) on the best and the current neighbor
    Neighbor* best;
    Neighbor* current;

    // true if the move is accepted
    bool isAccept ;
};


#endif
