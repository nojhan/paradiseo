// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoSelectors.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTORS_H_
#define MOEOSELECTORS_H_

#include <moeoComparator.h>


template <class It,class MOEOT>
It mo_deterministic_tournament(It _begin, It _end, unsigned _t_size,moeoComparator<MOEOT>& _comparator ,eoRng& _gen = rng)
{
    It best = _begin + _gen.random(_end - _begin);

    for (unsigned i = 0; i < _t_size - 1; ++i)
    {
        It competitor = _begin + _gen.random(_end - _begin);
        // compare the two individuals by using the comparator
        if (_comparator(*best, *competitor))
            // best "better" than competitor
            best=competitor;
    }
    return best;
}


template <class MOEOT>
const MOEOT& mo_deterministic_tournament(const eoPop<MOEOT>& _pop, unsigned _t_size,moeoComparator<MOEOT>& _comparator, eoRng& _gen = rng)
{
    return *mo_deterministic_tournament(_pop.begin(), _pop.end(),_t_size,_comparator, _gen);
}

template <class MOEOT>
MOEOT& mo_deterministic_tournament(eoPop<MOEOT>& _pop, unsigned _t_size,moeoComparator<MOEOT>& _comparator,eoRng& _gen = rng)
{
    return *mo_deterministic_tournament(_pop.begin(), _pop.end(), _t_size,_comparator, _gen);
}



template <class It,class MOEOT>
It mo_stochastic_tournament(It _begin, It _end, double _t_rate,moeoComparator<MOEOT>& _comparator ,eoRng& _gen = rng)
{
    It i1 = _begin + _gen.random(_end - _begin);
    It i2 = _begin + _gen.random(_end - _begin);

    bool return_better = _gen.flip(_t_rate);

    if (_comparator(*i1, *i2))
    {
        if (return_better) return i2;
        // else

        return i1;
    }
    else
    {
        if (return_better) return i1;
        // else
    }
    // else

    return i2;
}

template <class MOEOT>
const MOEOT& mo_stochastic_tournament(const eoPop<MOEOT>& _pop, double _t_rate,moeoComparator<MOEOT>& _comparator, eoRng& _gen = rng)
{
    return *mo_stochastic_tournament(_pop.begin(), _pop.end(), _t_rate,_comparator, _gen);
}

template <class MOEOT>
MOEOT& mo_stochastic_tournament(eoPop<MOEOT>& _pop, double _t_rate, eoRng& _gen = rng)
{
    return *mo_stochastic_tournament(_pop.begin(), _pop.end(), _t_rate, _gen);
}



template <class It>
It mo_roulette_wheel(It _begin, It _end, double total, eoRng& _gen = rng)
{

    float roulette = _gen.uniform(total);

    if (roulette == 0.0)	   // covers the case where total==0.0
        return _begin + _gen.random(_end - _begin); // uniform choice

    It i = _begin;

    while (roulette > 0.0)
    {
        roulette -= static_cast<double>(*(i++));
    }

    return --i;
}

template <class MOEOT>
const MOEOT& mo_roulette_wheel(const eoPop<MOEOT>& _pop, double total, eoRng& _gen = rng)
{
    float roulette = _gen.uniform(total);

    if (roulette == 0.0)	   // covers the case where total==0.0
        return _pop[_gen.random(_pop.size())]; // uniform choice

    typename eoPop<MOEOT>::const_iterator i = _pop.begin();

    while (roulette > 0.0)
    {
        roulette -= static_cast<double>((i++)->fitness());
    }

    return *--i;
}

template <class MOEOT>
MOEOT& mo_roulette_wheel(eoPop<MOEOT>& _pop, double total, eoRng& _gen = rng)
{
    float roulette = _gen.uniform(total);

    if (roulette == 0.0)	   // covers the case where total==0.0
        return _pop[_gen.random(_pop.size())]; // uniform choice

    typename eoPop<MOEOT>::iterator i = _pop.begin();

    while (roulette > 0.0)
    {
        // fitness only
        roulette -= static_cast<double>((i++)->fitness());
    }

    return *--i;
}

#endif /*MOEOSELECTORS_H_*/









