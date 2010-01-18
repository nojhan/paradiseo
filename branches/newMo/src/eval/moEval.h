#ifndef moEval_H
#define moEval_H

#include <eoFunctor.h>

template<class Neighbor>
class moEval : public eoBF<typename Neighbor::EOType &, Neighbor&, void>
{
public:
     typedef typename Neighbor::EOType EOT;
     typedef typename EOT::Fitness Fitness;
};

#endif
