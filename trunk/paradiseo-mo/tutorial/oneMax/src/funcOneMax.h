// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// funcOneMax.h
// 25/11/2009 : copy from FuncU.h
//-----------------------------------------------------------------------------

#ifndef __FuncOneMax
#define __FuncOneMax

template< class EOT >
class FuncOneMax : public eoEvalFunc<EOT>
{
private:
    unsigned int N;

public:
    FuncOneMax(unsigned int n) : N(n) {};

    ~FuncOneMax(void) {} ;

    void operator() (EOT & genome) {
        unsigned int sum = 0;

        for (unsigned int i = 0; i < N; i++)
            sum += genome[i];

        genome.fitness(sum);
    }

};

#endif
