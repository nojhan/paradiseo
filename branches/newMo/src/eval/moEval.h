#ifndef moEval_H
#define moEval_H

#include <eoFunctor.h>

/**
 * Abstract class for the evaluation
 */
template<class Neighbor>
class moEval : public eoBF<typename Neighbor::EOT &, Neighbor&, void>
{
public:
     typedef typename Neighbor::EOT EOT;
     typedef typename EOT::Fitness Fitness;
};

#endif
