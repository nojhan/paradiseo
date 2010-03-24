// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

#ifndef __funcRoyalRoad
#define __funcRoyalRoad

#include <eoEvalFunc.h>

template< class EOT >
class FuncRoyalRoad : public eoEvalFunc<EOT>
{
    // number of blocks
    unsigned n;

    // size of a block
    unsigned k;

public:
    FuncRoyalRoad(unsigned _n, unsigned _k) : n(_n), k(_k) {};

    ~FuncRoyalRoad(void) {} ;

    virtual void operator() (EOT & _solution)
    {
        unsigned sum = 0;
        unsigned i, j;

        for (i = 0; i < n; i++) {
            j = 0;
            while (_solution[i * n + j] && j < k) j++;

            if (j == k)
                sum++;
        }

        _solution.fitness(sum);
    };

};

#endif
