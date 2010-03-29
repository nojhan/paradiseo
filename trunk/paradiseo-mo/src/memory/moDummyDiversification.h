#ifndef _moDummyDiversification_h
#define _moDummyDiversification_h

#include <memory/moDiversification.h>
#include <memory/moDummyMemory.h>

/**
 * Dummy diversification strategy
 */
template< class Neighbor >
class moDummyDiversification : public moDiversification<Neighbor>, public moDummyMemory<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * @return always false
     */
    bool operator()(EOT &) {
        return false;
    }
};

#endif
