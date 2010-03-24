#ifndef _moMoveNeighborhood_h
#define _moMoveNeighborhood_h

#include <neighborhood/moMoveNeighbor.h>
#include <neighborhood/moNeighborhood.h>

#include <move/moMoveInit.h>
#include <move/moNextMove.h>

template< class M, class Fitness >
class moMoveNeighborhood : public moNeighborhood <moMoveNeighbor<M, Fitness> >
{
public:

    typedef moMoveNeighbor<M, Fitness> Neighbor;
    typedef typename M::EOType EOT;

    moMoveNeighborhood(moMoveInit<M>& i, moNextMove<M>& n):_init(i), _next(n), isContinue(true) {}

    virtual bool hasNeighbor(EOT & solution) {
        return true;
    }

    /*
      initialisation of the neighborhood
    */
    virtual void init(EOT & solution, Neighbor & current) {
        _init(*(current._move), solution);
        isContinue=true;
    }

    /*
    Give the next neighbor
    */
    virtual void next(EOT & solution, Neighbor & current) {
        isContinue=_next(*(current._move), solution);
    }

    /*
      if false, there is no neighbor left to explore
    */
    virtual bool cont(EOT & solution) {
        return isContinue;
    }

    /** Return the class id.
    *  @return the class name as a std::string
    */
    virtual std::string className() const {
        return "moMoveNeighborhood";
    }

private:
    moMoveInit<M>& _init;
    moNextMove<M>& _next;
    bool isContinue;

};


#endif
