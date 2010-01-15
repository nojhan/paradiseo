#ifndef _bitNeighbor_h
#define _bitNeighbor_h

#include <ga/eoBit.h>
#include <neighborhood/moNeighbor.h>

/*
  contener of the neighbor information
*/
template< class Fitness >
class moBitNeighbor : public moNeighbor< eoBit<Fitness> , Fitness>
{
public:
    typedef eoBit<Fitness> EOType ;

    using moNeighbor< eoBit<Fitness> , Fitness>::fitness;

    // describe the neighbor
    unsigned bit ;

    // empty constructor needed
    moBitNeighbor() : moNeighbor<eoBit<Fitness> , Fitness>() { } ;

    // copy constructor
    moBitNeighbor(const moBitNeighbor & n) : moNeighbor<eoBit<Fitness> , Fitness>(n) { 
	this->bit = n.bit ;
    } ;

    moBitNeighbor(unsigned b) : moNeighbor<eoBit<Fitness> , Fitness>() , bit(b) { } ;

    /*
    * operator of assignment
    */
    virtual moBitNeighbor<Fitness> & operator=(const moBitNeighbor<Fitness> & source) {
	moNeighbor<EOType, Fitness>::operator=(source);

	this->bit = source.bit ;

	return *this ;
    }

    /*
      move the solution
    */
    virtual void move(EOType & solution) {
	solution[bit] = solution[bit]?false:true ;
    }

    // by default: if the fitness of the current solution is stricly higher than the other neighbor
    virtual bool betterThan(const moNeighbor<EOType, Fitness> & __neighbor) {
	return (this->fitness() > __neighbor.fitness()) ;
    };

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moBitNeighbor"; }
    
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
	    Fitness repFit ;
	    _is.seekg(pos); // rewind
	    _is >> repFit;
	    _is >> bit;

	    fitness(repFit);
	}
    }

    /**
    * Write object. Called printOn since it prints the object _on_ a stream.
    * @param _os A std::ostream.
    */
    virtual void printOn(std::ostream& _os) const {
	_os << fitness() << ' ' << bit << ' ' ;
    }
    

};

#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
