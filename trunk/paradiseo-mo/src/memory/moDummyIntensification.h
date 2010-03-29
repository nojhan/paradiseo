#ifndef _moDummyIntensification_h
#define _moDummyIntensification_h

#include <memory/moIntensification.h>
#include <memory/moDummyMemory.h>

/**
 * Dummy intensification strategy
 */
template< class Neighbor >
class moDummyIntensification : public moIntensification<Neighbor>, public moDummyMemory<Neighbor>
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
