/**

 *	Pool allocator for the subtree and parse tree classes (homebrew and not compliant to ANSI allocator requirements)
 *  (c) copyright Maarten Keijzer 1999, 2000

 * Permission to copy, use,  modify, sell and distribute this software is granted provided
 * this copyright notice appears in all copies. This software is provided "as is" without
 * express or implied warranty, and with no claim as to its suitability for
 * any purpose.

 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.


*/

#ifndef node_pool_h
#define node_pool_h

class MemPool
{
public :

    MemPool(unsigned int sz) : esize(sz<sizeof(Link)? sizeof(Link) : sz) {}
    ~MemPool()
    {
        Chunk* n = chunks;
        while(n)
        {
            Chunk* p = n;
            n = n->next;
            delete p;
        }
    }

    void* allocate()
    {
        if (head == 0) grow();
        Link* p = head;
        head = p->next;
        return static_cast<void*>(p);
    }

    void deallocate(void* b)
    {
        Link* p = static_cast<Link*>(b);
        p->next = head;
        head = p;
    }

private :

    void grow()
    {
        Chunk* n = new Chunk;
        n->next = chunks;
        chunks = n;

        const int nelem = Chunk::size/esize;
        char* start = n->mem;
        char* last  = &start[(nelem-1)*esize];
        for (char* p = start; p < last; p += esize)
        {
            reinterpret_cast<Link*>(p)->next =
                reinterpret_cast<Link*>(p + esize);
        }

        reinterpret_cast<Link*>(last)->next = 0;
        head = reinterpret_cast<Link*>(start);
    }

    struct Link
    {
        Link* next;
    };

    struct Chunk
    {
        enum {size = 8 * 1024 - 16};
        Chunk* next;
        char mem[size];
    };

    Chunk* chunks;
    const unsigned int esize;
    Link* head;
};

template<class T>
class Node_alloc
{
public :

    T* allocate(void)
    {
        T* t = static_cast<T*>(mem.allocate());
        t = new  (t) T;
        return t;
    }

    T* construct(const T& org)
    {
        T* t = static_cast<T*>(mem.allocate());
        t = new  (t) T(org);
        return t;
    }

    void deallocate(T* t)
    {
        t->~T(); // call destructor
        mem.deallocate(static_cast<void*>(t));
    }

private :
    static MemPool mem;
};


template <class T>
class Standard_alloc
{
public :
    Standard_alloc() {}

    T* allocate(size_t arity = 1)
    {
        if (arity == 0)
            return 0;

        return new T [arity];
    }

    T* construct(size_t arity, T* org)
    {
        if (arity == 0)
            return 0;

        T* t = new T [arity];

        for (int i = 0; i < arity; ++i)
        {
            t = T(org[i]);
        }
    }

    void deallocate(T* t, size_t arity = 1)
    {
        if (arity == 0)
            return ;

        delete [] t;
    }

};

template <class T>
class Standard_Node_alloc
{
public :
    Standard_Node_alloc() {}

    T* allocate(void)
    {
        return new T;// [arity];
    }

    T* construct(const T& org)
    {
        return new T(org);
    }

    void deallocate(T* t)
    {
        delete t;
    }

};

template <class T>
class Tree_alloc
{
public :
    Tree_alloc() {}

    T* allocate(size_t arity)
    {
        T* t;

        switch(arity)
        {

        case 0 : return 0;
        case 1 :
            {
                t = static_cast<T*>(mem1.allocate());
                new (t) T;
                break;
            }
        case 2 :
            {
                t = static_cast<T*>(mem2.allocate());
                new (t) T;
                new (&t[1]) T;
                break;
            }
        case 3 :
            {
                t = static_cast<T*>(mem3.allocate());
                new (t) T;
                new (&t[1]) T;
                new (&t[2]) T;
                break;
            }
        default :
            {
                return new T[arity];
            }
        }

        return t;
     }

    T* construct(size_t arity, T* org)
    {
        T* t;

        switch(arity)
        {

        case 0 : return 0;
        case 1 :
            {
                t = static_cast<T*>(mem1.allocate());
                new (t) T(*org);
                break;
            }
        case 2 :
            {
                t = static_cast<T*>(mem2.allocate());
                new (t) T(*org);
                new (&t[1]) T(org[1]);
                break;
            }
        case 3 :
            {
                t = static_cast<T*>(mem3.allocate());
                new (t) T(*org);
                new (&t[1]) T(org[1]);
                new (&t[1]) T(org[2]);
                break;
            }
        default :
            {
                t = new T[arity]; // does call default ctor
                for (int i = 0; i < arity; ++i)
                {
                    t[i] = T(org[i]); // constructs now
                }
            }
        }

        return t;
     }



    void deallocate(T* t, size_t arity)
    {
        switch(arity)
        {
        case 0: return;
        case 3 :
            {
                t[2].~T(); t[1].~T(); t[0].~T();
                mem3.deallocate(static_cast<void*>(t));
                return;
            }
        case 2 :
            {
                t[1].~T(); t[0].~T();
                mem2.deallocate(static_cast<void*>(t));
                return;
            }
        case 1 :
            {
                t[0].~T();
                mem1.deallocate(static_cast<void*>(t));
                return;
            }
        default :
            {
                delete [] t;
                return;
            }
        }
    }


private :
    static MemPool mem1;
    static MemPool mem2;
    static MemPool mem3;
};

// static (non thread_safe) memory pools
template <class T> MemPool Node_alloc<T>::mem  = sizeof(T);

template <class T> MemPool Tree_alloc<T>::mem1 = sizeof(T);
template <class T> MemPool Tree_alloc<T>::mem2 = sizeof(T) * 2;
template <class T> MemPool Tree_alloc<T>::mem3 = sizeof(T) * 3;

#endif
