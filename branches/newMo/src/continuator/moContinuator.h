#ifndef _moContinuator_h
#define _moContinuator_h

#include <eoFunctor.h>

/**
 * To make specific continuator from a solution
 */
template< class NH >
class moContinuator : public eoUF<typename NH::EOT &, bool>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;

    /**
     * Init Continuator parameters
     * @param _solution the related solution
     */
    virtual void init(EOT& _solution) = 0 ;
};

#endif
