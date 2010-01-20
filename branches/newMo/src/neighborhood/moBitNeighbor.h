#ifndef _bitNeighbor_h
#define _bitNeighbor_h

#include <ga/eoBit.h>
#include <neighborhood/moBackableNeighbor.h>

/**
 * Neighbor related to a vector of Bit
 */
template< class Fitness >
class moBitNeighbor : public moBackableNeighbor<eoBit<Fitness>, Fitness>
{
public:
    typedef eoBit<Fitness> EOT ;

    using moNeighbor<eoBit<Fitness>, Fitness>::fitness;

    // describe the neighbor
    unsigned bit ;

    /**
     * Default Constructor
     */
    moBitNeighbor() : moBackableNeighbor<eoBit<Fitness> , Fitness>() { } ;

    /**
     * Copy Constructor
     */
    moBitNeighbor(const moBitNeighbor& _n) : moNeighbor<eoBit<Fitness> , Fitness>(_n) {
    	this->bit = _n.bit ;
    } ;

    /**
     * Constructor
     * @param _b index
     */
    moBitNeighbor(unsigned int _b) : moNeighbor<eoBit<Fitness> , Fitness>() , bit(_b) { } ;

    /**
     * Assignment operator
     */
    virtual moBitNeighbor<Fitness> & operator=(const moBitNeighbor<Fitness> & _source) {
    	moNeighbor<EOT, Fitness>::operator=(_source);
    	this->bit = _source.bit ;
    	return *this ;
    }

    /**
     * move the solution
     * @param _solution the solution to move
     */
    virtual void move(EOT & _solution) {
    	_solution[bit] = !_solution[bit];
    }

    /**
     * move back the solution (useful for the evaluation by modif)
     * @param _solution the solution to move back
     */
    virtual void moveBack(EOT & _solution) {
    	_solution[bit] = !_solution[bit];
    }

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
    	return "moBitNeighbor";
    }
    
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
		if (fitness_str == "INVALID"){
			throw std::runtime_error("invalid fitness");
		}
		else{
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
    	_os << fitness() << ' ' << bit << std::endl;
    }
    

};

#endif
