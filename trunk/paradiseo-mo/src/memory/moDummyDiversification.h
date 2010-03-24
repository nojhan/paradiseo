#ifndef _moDummyDiversification_h
#define _moDummyDiversification_h

#include <memory/moDiversification.h>

/**
 * Dummy diversification strategy
 */
template< class Neighbor >
class moDummyDiversification : public moDiversification<Neighbor>
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
