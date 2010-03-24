#ifndef _moDummyIntensification_h
#define _moDummyIntensification_h

#include <memory/moIntensification.h>

/**
 * Dummy intensification strategy
 */
template< class Neighbor >
class moDummyIntensification : public moIntensification<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Init : NOTHIING TO DO
     */
    void init(EOT & _sol) {}

    /**
     * Add : NOTHIING TO DO
     */
    void add(EOT & _sol, Neighbor & _neighbor) {}

    /**
     * Update : NOTHIING TO DO
     */
    void update(EOT & _sol, Neighbor & _neighbor) {}

    /**
     * ClearMemory : NOTHIING TO DO
     */
    void clearMemory() {}

    /**
     * @return always false
     */
    bool operator()(EOT &) {
        return false;
    }
};

#endif
