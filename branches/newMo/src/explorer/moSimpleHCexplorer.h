#ifndef _moSimpleHCexplorer_h
#define _moSimpleHCexplorer_h

#include <explorer/moNeighborhoodExplorer.h>

template< class Neighborhood >
class moSimpleHCexplorer : public moNeighborhoodExplorer<Neighborhood>
{
public:
    typedef typename Neighborhood::EOT EOT ;
    typedef typename Neighborhood::Neighbor Neighbor ;

    using moNeighborhoodExplorer<Neighborhood>::neighborhood;
    using moNeighborhoodExplorer<Neighborhood>::eval;
    using moNeighborhoodExplorer<Neighborhood>::comparator;

    // empty constructor
    moSimpleHCexplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _comparator) : moNeighborhoodExplorer<Neighborhood>(_neighborhood, _eval, _comparator){
    	isAccept = false;
    	current=new Neighbor();
    	best=new Neighbor();
    }

    virtual void initParam (EOT & solution) { } ;

    virtual void updateParam (EOT & solution) { } ;

    virtual void terminate (EOT & solution) { } ;

    virtual void operator() (EOT & solution) {

	//est qu'on peut initializer

		if(neighborhood.hasNeighbor(solution)){
			neighborhood.init(solution, (*current));

			eval(solution, (*current));

			std::cout <<"sol et neighbor:"<< solution << ", "<< (*current) << std::endl;

			(*best) = (*current);

			while (neighborhood.cont(solution)) {
				neighborhood.next(solution, (*current));

				eval(solution, (*current));
				std::cout <<"sol et neighbor:"<< solution << ", "<< (*current) << std::endl;
				if (comparator((*current), (*best))) {
					(*best) = (*current);
				}
			}
		}
		else{
			isAccept=false;
		}
    };

    virtual bool isContinue(EOT & solution) {
    	return isAccept ;
    };

    virtual void move(EOT & solution) {
		
    	(*best).move(solution);

    	solution.fitness((*best).fitness());
    };

    virtual bool accept(EOT & solution) {	
		if(neighborhood.hasNeighbor(solution)){
			isAccept = (solution.fitness() < (*best).fitness()) ;
		}
		return isAccept;
    };

private:

// attention il faut que le constructeur vide existe
    Neighbor* best;

    Neighbor* current;

    // true if the move is accepted
    bool isAccept ;
};


#endif
