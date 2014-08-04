/*
* <moeoSelectors.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTORS_H_
#define MOEOSELECTORS_H_

#include "../comparator/moeoComparator.h"


template <class It,class MOEOT>
It mo_deterministic_tournament(It _begin, It _end, unsigned int _t_size,moeoComparator<MOEOT>& _comparator ,eoRng& _gen = rng)
{
  It best = _begin + _gen.random(_end - _begin);

  for (unsigned int i = 0; i < _t_size - 1; ++i)
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
const MOEOT& mo_deterministic_tournament(const eoPop<MOEOT>& _pop, unsigned int _t_size,moeoComparator<MOEOT>& _comparator, eoRng& _gen = rng)
{
  return *mo_deterministic_tournament(_pop.begin(), _pop.end(),_t_size,_comparator, _gen);
}


template <class MOEOT>
MOEOT& mo_deterministic_tournament(eoPop<MOEOT>& _pop, unsigned int _t_size,moeoComparator<MOEOT>& _comparator,eoRng& _gen = rng)
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









