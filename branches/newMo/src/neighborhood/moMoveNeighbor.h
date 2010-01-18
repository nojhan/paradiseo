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
	moMoveNeighbor() {
	  move=new M();
	};

	~moMoveNeighbor() {
	  delete move;
	};

	// copy constructeur
	moMoveNeighbor(const moMoveNeighbor<M, Fitness> & _n) {
	  moNeighbor<EOT, Fitness>::operator=(_n);
	  (*move) = *(_n.getMove());
	}
 
    // assignment operator
    virtual moMoveNeighbor<M, Fitness> & operator=(const moMoveNeighbor<M, Fitness> & _n) {
      moNeighbor <EOT, Fitness>::operator=(_n);
      (*move) = *(_n.getMove());
      return *this ;
    }
 
    /*
    * move the solution
    */
    virtual void move(EOT & _solution){
      (*move)(_solution);
    }

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const { return "moMoveNeighbor"; }
    
    void setMove(M* _move){
    	move=_move;
    }

    M* getMove(){
    	return move;
    }

    
private:
    M* move;
    
};

#endif
