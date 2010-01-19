#ifndef _moNeighbor_h
#define _moNeighbor_h

//EO inclusion
#include <EO.h>
#include <eoObject.h>
#include <eoPersistent.h>

#include <comparator/moNeighborComparator.h>

/*
 * Container of the neighbor informations
 */
template< class EOT , class Fitness >
class moNeighbor : public eoObject, public eoPersistent
{
public:

    /*
     * Default Constructor
     */
    moNeighbor(){}

    /*
     * Copy Constructor
     * @param _neighbor to copy
     */
    moNeighbor(const moNeighbor<EOType, Fitness>& _neighbor) {
    	repFitness = _neighbor.fitness();
    }
 
    /*
     * Assignment operator
     * @param the _neighbor to assign
     * @return a neighbor equal to the other
     */
    virtual moNeighbor<EOT, Fitness>& operator=(const moNeighbor<EOT, Fitness>& _neighbor) {
    	repFitness = _neighbor.fitness();
    	return (*this);
    }
 
    /*
     * Move a solution
     * @param _solution the related solution
     */
    virtual void move(EOT & _solution) = 0 ;

    /*
     * Get the fitness of the neighbor
     * @return fitness of the neighbor
     */
    const Fitness& fitness() const {
    	return repFitness;
    }
    

    /*
     * Get fitness as reference, useful when fitness is set in a multi-stage way, e.g., MOFitness gets performance information, is subsequently ranked
     * @return fitness as reference of the neighbor
     */
    Fitness& fitnessReference() {
    	return repFitness;
    }

    /*
     * Set fitness. At the same time, validates it.
     * @param _fitness new fitness value.
     */
    void fitness(const Fitness& _fitness){
	    repFitness = _fitness;
	}
    
    /*
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const { return "moNeighbor"; }
    
    /*
     * Read object.
     * Calls base class, just in case that one had something to do.
     * The read and print methods should be compatible and have the same format.
     * In principle, format is "plain": they just print a number
     * @param _is a std::istream.
     */
    virtual void readFrom(std::istream& _is) {
		_is >> repFitness;
    }

    /*
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
