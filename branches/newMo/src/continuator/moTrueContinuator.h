#ifndef _moTrueContinuator_h
#define _moTrueContinuator_h

#include <continuator/moContinuator.h>

/**
 * Continuator always return True
 */
template< class NH >
class moTrueContinuator : public moContinuator<NH>
{
public:
    typedef typename NH::EOT EOT ;

    // empty constructor
    moTrueContinuator() {} ;

    /**
     *@param _solution a solution
     *@return always true
     */
    virtual bool operator()(EOT & _solution) {
    	return true;
    }

    /**
     * NOTHING TO DO
     * @param _solution a solution
     */
    virtual void init(EOT & _solution) {}

};

#endif
