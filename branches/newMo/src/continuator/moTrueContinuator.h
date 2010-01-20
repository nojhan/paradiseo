#ifndef _moTrueContinuator_h
#define _moTrueContinuator_h

#include <continuator/moContinuator.h>

/**
  to make specific continuator from a solution
*/
template< class NH >
class moTrueContinuator : public moContinuator<NH>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;

    // empty constructor
    moTrueContinuator() { i=0;} ;

    /**
    always true
    */
    virtual bool operator()(EOT & solution) {
      i++;
	return i<10;
    };

    virtual void init(EOT & solution) {
    }

    int i;
};

#endif
