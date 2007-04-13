// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

#ifndef _MOEOMOVEINCREVAL_H
#define _MOEOMOVEINCREVAL_H

#include <eoFunctor.h>

template < class Move >
class moeoMoveIncrEval : public eoBF < const Move &, const typename Move::EOType &, typename Move::EOType::ObjectiveVector >
    {};

#endif
