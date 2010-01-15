#ifndef _moSimpleHCexplorer_h
#define _moSimpleHCexplorer_h

#include <explorer/moNeighborhoodExplorer.h>

template< class NH >
class moSimpleHCexplorer : public moNeighborhoodExplorer<NH>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;
    typedef typename Neighborhood::Neighbor Neighbor ;

    // empty constructor
    moSimpleHCexplorer(Neighborhood & __neighborhood) : neighborhood(__neighborhood) {  
	isAccept = false; 
    }

    virtual void initParam (EOT & solution) { } ;

    virtual void updateParam (EOT & solution) { } ;

    virtual void terminate (EOT & solution) { } ;

    virtual void operator() (EOT & solution) {

//est qu'on peut initializer
	
	if(neighborhood.hasNeighbor(solution)){
	    neighborhood.init(solution, current);
	    
	    current.eval(solution);
	    
	    best = current;

	    while (neighborhood.cont(solution)) {
		neighborhood.next(solution, current);

		current.eval(solution);
		
		if (current.betterThan(best)) {
		    
		    best = current;
		    
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
		
	best.move(solution);

	solution.fitness(best.fitness());
    };

    virtual bool accept(EOT & solution) {	
	if(neighborhood.hasNeighbor(solution)){
	    isAccept = (solution.fitness() < best.fitness()) ;	    
	}
	return isAccept;
    };

private:
    Neighborhood & neighborhood;

// attention il faut que le constructeur vide existe
    Neighbor best ;

    Neighbor current ;

    // true if the move is accepted
    bool isAccept ;
};


#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
