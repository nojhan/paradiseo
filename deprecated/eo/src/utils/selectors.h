/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  selectors.h
    A bunch of useful selector functions. They generally have three forms:

    template <class It>
    It select(It begin, It end, params, eoRng& gen = rng);

    template <class EOT>
    const EOT& select(const eoPop<EOT>& pop, params, eoRng& gen = rng);

    template <class EOT>
    EOT& select(eoPop<EOT>& pop, params, eoRng& gen = rng);

    where select is one of: roulette_wheel, deterministic_tournament
    and stochastic_tournament (at the moment).

 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */

#ifndef SELECT__H
#define SELECT__H

#include <stdexcept>

#include "eoRNG.h"
#include <eoPop.h>
/**
@addtogroup Selectors
@{
*/

template <class EOT>
bool minimizing_fitness()
{
    EOT eo1; // Assuming people don't do anything fancy in the default constructor!
    EOT eo2;

    /* Dear user, when the two line below do not compile you are most
       likely not working with scalar fitness values. In that case we're sorry
       but you cannot use lottery or roulette_wheel selection...
    */

#ifdef _MSC_VER
    eo1.fitness( EOT::Fitness(0.0) );
    eo2.fitness( EOT::Fitness(1.0) );
#else
    eo1.fitness( typename EOT::Fitness(0.0) ); // tried to cast it to an EOT::Fitness, but for some reason GNU barfs on this
    eo2.fitness( typename EOT::Fitness(1.0) );
#endif

    return eo2 < eo1; // check whether we have a minimizing fitness
}

inline double scale_fitness(const std::pair<double, double>& _minmax, double _value)
{
    if (_minmax.first == _minmax.second)
    {
        return 0.0; // no differences in fitness, population converged!
    }
    // else

    return (_value - _minmax.first) / (_minmax.second - _minmax.first);
}

template <class It>
double sum_fitness(It begin, It end)
{
    double sum = 0.0;

    for (; begin != end; ++begin)
    {
        double v = static_cast<double>(begin->fitness());
        if (v < 0.0)
            throw std::logic_error("sum_fitness: negative fitness value encountered");
        sum += v;
    }

    return sum;
}

template <class EOT>
double sum_fitness(const eoPop<EOT>& _pop)
{
    return sum_fitness(_pop.begin(), _pop.end());
}

template <class EOT>
double sum_fitness(const eoPop<EOT>& _pop, std::pair<double, double>& _minmax)
{
    double rawTotal, scaledTotal;

    typename eoPop<EOT>::const_iterator it = _pop.begin();

    _minmax.first = it->fitness();
    _minmax.second = it++->fitness();

    for(; it != _pop.end(); ++it)
    {
        double v = static_cast<double>(it->fitness());

        _minmax.first = std::min(_minmax.first, v);
        _minmax.second = std::max(_minmax.second, v);

        rawTotal += v;
    }

    if (minimizing_fitness<EOT>())
    {
        std::swap(_minmax.first, _minmax.second);
    }

    scaledTotal = 0.0;

    // unfortunately a second loop is neccessary to scale the fitness
    for (it = _pop.begin(); it != _pop.end(); ++it)
    {
        double v = scale_fitness(_minmax, static_cast<double>(it->fitness()));

        scaledTotal += v;
    }

    return scaledTotal;
}

template <class It>
It roulette_wheel(It _begin, It _end, double total, eoRng& _gen = rng)
{

    double roulette = _gen.uniform(total);

    if (roulette == 0.0)           // covers the case where total==0.0
      return _begin + _gen.random(_end - _begin); // uniform choice

    It i = _begin;

    while (roulette > 0.0)
    {
            roulette -= static_cast<double>(*(i++));
    }

    return --i;
}

template <class EOT>
const EOT& roulette_wheel(const eoPop<EOT>& _pop, double total, eoRng& _gen = rng)
{
    double roulette = _gen.uniform(total);

    if (roulette == 0.0)           // covers the case where total==0.0
      return _pop[_gen.random(_pop.size())]; // uniform choice

    typename eoPop<EOT>::const_iterator i = _pop.begin();

    while (roulette > 0.0)
    {
            roulette -= static_cast<double>((i++)->fitness());
    }

    return *--i;
}

template <class EOT>
EOT& roulette_wheel(eoPop<EOT>& _pop, double total, eoRng& _gen = rng)
{
    float roulette = _gen.uniform(total);

    if (roulette == 0.0)           // covers the case where total==0.0
      return _pop[_gen.random(_pop.size())]; // uniform choice

    typename eoPop<EOT>::iterator i = _pop.begin();

    while (roulette > 0.0)
    {
            roulette -= static_cast<double>((i++)->fitness());
    }

    return *--i;
}

template <class It>
It deterministic_tournament(It _begin, It _end, unsigned _t_size, eoRng& _gen = rng)
{
    It best = _begin + _gen.random(_end - _begin);

    for (unsigned i = 0; i < _t_size - 1; ++i)
    {
        It competitor = _begin + _gen.random(_end - _begin);

        if (*best < *competitor)
        {
            best = competitor;
        }
    }

    return best;
}

template <class EOT>
const EOT& deterministic_tournament(const eoPop<EOT>& _pop, unsigned _t_size, eoRng& _gen = rng)
{
    return *deterministic_tournament(_pop.begin(), _pop.end(), _t_size, _gen);
}

template <class EOT>
EOT& deterministic_tournament(eoPop<EOT>& _pop, unsigned _t_size, eoRng& _gen = rng)
{
    return *deterministic_tournament(_pop.begin(), _pop.end(), _t_size, _gen);
}

template <class It>
It inverse_deterministic_tournament(It _begin, It _end, unsigned _t_size, eoRng& _gen = rng)
{
    It worst = _begin + _gen.random(_end - _begin);

    for (unsigned i = 1; i < _t_size; ++i)
    {
        It competitor = _begin + _gen.random(_end - _begin);

        if (competitor == worst)
        {
            --i;
            continue; // try again
        }

        if (*competitor < *worst)
        {
            worst = competitor;
        }
    }

    return worst;
}

template <class EOT>
const EOT& inverse_deterministic_tournament(const eoPop<EOT>& _pop, unsigned _t_size, eoRng& _gen = rng)
{
    return *inverse_deterministic_tournament<EOT>(_pop.begin(), _pop.end(), _t_size, _gen);
}

template <class EOT>
EOT& inverse_deterministic_tournament(eoPop<EOT>& _pop, unsigned _t_size, eoRng& _gen = rng)
{
    return *inverse_deterministic_tournament(_pop.begin(), _pop.end(), _t_size, _gen);
}

template <class It>
It stochastic_tournament(It _begin, It _end, double _t_rate, eoRng& _gen = rng)
{
  It i1 = _begin + _gen.random(_end - _begin);
  It i2 = _begin + _gen.random(_end - _begin);

  bool return_better = _gen.flip(_t_rate);

  if (*i1 < *i2)
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

template <class EOT>
const EOT& stochastic_tournament(const eoPop<EOT>& _pop, double _t_rate, eoRng& _gen = rng)
{
    return *stochastic_tournament(_pop.begin(), _pop.end(), _t_rate, _gen);
}

template <class EOT>
EOT& stochastic_tournament(eoPop<EOT>& _pop, double _t_rate, eoRng& _gen = rng)
{
    return *stochastic_tournament(_pop.begin(), _pop.end(), _t_rate, _gen);
}

template <class It>
It inverse_stochastic_tournament(It _begin, It _end, double _t_rate, eoRng& _gen = rng)
{
  It i1 = _begin + _gen.random(_end - _begin);
  It i2 = _begin + _gen.random(_end - _begin);

  bool return_worse = _gen.flip(_t_rate);

  if (*i1 < *i2)
  {
      if (return_worse) return i1;
      // else

      return i2;
   }
    else
    {
      if (return_worse) return i2;
      // else
    }
    // else

    return i1;
}

template <class EOT>
const EOT& inverse_stochastic_tournament(const eoPop<EOT>& _pop, double _t_rate, eoRng& _gen = rng)
{
    return *inverse_stochastic_tournament(_pop.begin(), _pop.end(), _t_rate, _gen);
}

template <class EOT>
EOT& inverse_stochastic_tournament(eoPop<EOT>& _pop, double _t_rate, eoRng& _gen = rng)
{
    return *inverse_stochastic_tournament(_pop.begin(), _pop.end(), _t_rate, _gen);
}

/** @} */

#endif
