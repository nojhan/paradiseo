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
    * make the evaluation of the current neighbor and update the information on this neighbor
    * the evaluation could be increamental
    */
    virtual void eval(EOT & solution) = 0 ;

    /*
    * move the solution
    */
    virtual void move(EOT & solution) = 0 ;

    // true if the this is better than the neighbor __neighbor
//    virtual bool betterThan(const moNeighbor<EOT,Fitness> & __neighbor) = 0 ;
    virtual bool betterThan(const moNeighbor<EOT,Fitness> & __neighbor) {
	return (*neighborComparator)(*this, __neighbor) ;
    } ;

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
    * @throw runtime_std::exception If a valid object can't be read.
    */
    virtual void readFrom(std::istream& _is) {
			std::string fitness_str;
			int pos = _is.tellg();
		_is >> fitness_str;

		if (fitness_str == "INVALID")
		{
			throw std::runtime_error("invalid fitness");
		}
		else
		{
			_is.seekg(pos); // rewind
			_is >> repFitness;
		}
    }

    /**
    * Write object. Called printOn since it prints the object _on_ a stream.
    * @param _os A std::ostream.
    */
    virtual void printOn(std::ostream& _os) const {
		_os << repFitness << ' ' ;
    }
    
    static void setNeighborComparator(const moNeighborComparator< moNeighbor<EOType, Fitness> > & comparator) {
    	neighborComparator = & comparator ;
    }

    static const moNeighborComparator< moNeighbor<EOType, Fitness> > & getNeighborComparator() {
    	return *neighborComparator ;
    }

private:
    // minimal information on the neighbor : fitness
    Fitness repFitness ;

    // the comparator of neighbors
    static moNeighborComparator<moNeighbor<EOType, Fitness> > * neighborComparator ;

};

// static default comparor 
template<class EOT, class Fitness>
moNeighborComparator<moNeighbor<EOT, Fitness> > * moNeighbor<EOT, Fitness>::neighborComparator = new  moNeighborComparator<moNeighbor<EOT, Fitness> >();


#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
