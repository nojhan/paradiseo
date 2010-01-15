#ifndef _moMoveNeighbor_h
#define _moMoveNeighbor_h

#include <eo>

#include <neighborhood/moNeighbor.h>
#include <move/moMoveIncrEval.h>
#include <move/moMove.h>


/*
  contener of the neighbor informations
*/
template< class M , class Fitness >
  class moMoveNeighbor : public moNeighbor <typename M::EOType, Fitness>
{
public:
  
  typedef typename M::EOType EOT;

    // empty constructor
  moMoveNeighbor() {_move=new M();};

  ~moMoveNeighbor() {delete _move;};

    // copy constructeur
    moMoveNeighbor(const moMoveNeighbor<M, Fitness> & _n) {
      moNeighbor<EOT, Fitness>::operator=(_n);
	(*_move) = *(_n._move);
    }
 
    // assignment operator
    virtual moMoveNeighbor<M, Fitness> & operator=(const moMoveNeighbor<M, Fitness> & _n) {
      moNeighbor <EOT, Fitness>::operator=(_n);
	(*_move) = *(_n._move);

      std::cout << moNeighbor<EOT, Fitness>::fitness() << " , " << _n.fitness() << std::endl;
      return *this ;
    }
 
    /*
    * make the evaluation of the current neighbor and update the information on this neighbor
    * the evaluation could be increamental
    */
    virtual void eval(EOT & solution){
	fitness((*_incrEval)(*_move, solution));
    }

    /*
    * move the solution
    */
    virtual void move(EOT & solution){
      (*_move)(solution);
    }

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moMoveNeighbor"; }
    
    static void setIncrEval(moMoveIncrEval<M, Fitness>& increm) {
	_incrEval = & increm ;
    }

    /**
    * Read object.\							\
    * Calls base class, just in case that one had something to do.
    * The read and print methods should be compatible and have the same format.
    * In principle, format is "plain": they just print a number
    * @param _is a std::istream.
    * @throw runtime_std::exception If a valid object can't be read.
    */
    /*    virtual void readFrom(std::istream& _is) {
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
*/

    /**
    * Write object. Called printOn since it prints the object _on_ a stream.
    * @param _os A std::ostream.
    */
    /*virtual void printOn(std::ostream& _os) const {
	_os << repFitness << ' ' ;
	}*/

    M* _move;
    
private:

    static moMoveIncrEval<M, Fitness>* _incrEval;
    
};

template< class M , class Fitness >
  moMoveIncrEval<M, Fitness> * moMoveNeighbor<M, Fitness>::_incrEval = NULL;
#endif
