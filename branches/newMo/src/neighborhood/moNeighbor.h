#ifndef _moNeighbor_h
#define _moNeighbor_h

#include <eo>

#include <comparator/moNeighborComparator.h>

/*
  contener of the neighbor informations
*/
template< class EOT , class Fitness >
class moNeighbor : public eoObject, public eoPersistent
{
public:
    typedef EOT EOType ;

    // empty constructor
    moNeighbor() {  } ;

    // copy constructeur
    moNeighbor(const moNeighbor<EOType, Fitness> & _n) {
    	repFitness = _n.fitness();
    }
 
    // assignment operator
    virtual moNeighbor<EOType, Fitness> & operator=(const moNeighbor<EOType, Fitness> & _n) {
    	repFitness = _n.fitness();
    	return *this ;
    }
 
    /*
    * move the solution
    */
    virtual void move(EOT & solution) = 0 ;

    /// Return fitness value.
    const Fitness& fitness() const {
    	return repFitness;
    }
    
    /// Get fitness as reference, useful when fitness is set in a multi-stage way, e.g., MOFitness gets performance information, is subsequently ranked
    Fitness& fitnessReference() {
    	return repFitness;
    }

    /** Set fitness. At the same time, validates it.
    *  @param _fitness New fitness value.
    */
    void fitness(const Fitness& _fitness)
	{
	    repFitness = _fitness;
	}
    
    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moNeighbor"; }
    
    /**
    * Read object.\							\
    * Calls base class, just in case that one had something to do.
    * The read and print methods should be compatible and have the same format.
    * In principle, format is "plain": they just print a number
    * @param _is a std::istream.
    */
    virtual void readFrom(std::istream& _is) {
			_is >> repFitness;
    }

    /**
    * Write object. Called printOn since it prints the object _on_ a stream.
    * @param _os A std::ostream.
    */
    virtual void printOn(std::ostream& _os) const {
		_os << repFitness << ' ' ;
    }
    
private:
    // minimal information on the neighbor : fitness
    Fitness repFitness ;

};

#endif
