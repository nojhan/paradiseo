#ifndef _moMemory_h
#define _moMemory_h

/**
 * Abstract class for different memory
 */
template< class Neighbor >
class moMemory //: public eoObject
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Init the memory
     * @param _sol the current solution
     */
    virtual void init(EOT & _sol) = 0;

    /**
     * Add data to the memory
     * @param _sol the current solution
     * @param _neighbor the current neighbor
     */
    virtual void add(EOT & _sol, Neighbor & _neighbor) = 0;

    /**
     * update the memory
     * @param _sol the current solution
     * @param _neighbor the current neighbor
     */
    virtual void update(EOT & _sol, Neighbor & _neighbor) = 0;

    /**
     * clear the memory
     */
    virtual void clearMemory() = 0;

};

#endif
