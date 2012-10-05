// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRandomSelect.h
// (c) GeNeura Team, 1998 - EEAAX 1999, Maarten Keijzer 2000
/*
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
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */

#ifndef eoRandomSelect_h
#define eoRandomSelect_h

/** This file contains straightforward selectors:
 * eoRandomSelect          returns an individual uniformly selected
 * eoBestSelect            always return the best individual
 * eoSequentialSelect      returns all individuals in turn
 */

#include <utils/eoRNG.h>
#include <eoSelectOne.h>

/** eoRandomSelect: a selection method that selects ONE individual randomly
 *
 * @ingroup Selectors
 */
template <class EOT> class eoRandomSelect: public eoSelectOne<EOT>
{
 public:

  /// not a big deal!!!
  virtual const EOT& operator()(const eoPop<EOT>& _pop)
  {
    return _pop[eo::rng.random(_pop.size())] ;
  }
};

/** eoBestSelect: a selection method that always return the best
 *                (mainly for testing purposes)
 *
 * @ingroup Selectors
 */
template <class EOT> class eoBestSelect: public eoSelectOne<EOT>
{
 public:

  /// not a big deal!!!
  virtual const EOT& operator()(const eoPop<EOT>& _pop)
  {
    return _pop.best_element() ;
  }
};

/** eoNoSelect: returns all individual in order WITHOUT USING FITNESS!!!
 *       looping back to the beginning when exhasuted
 *
 * @ingroup Selectors
 */
template <class EOT> class eoNoSelect: public eoSelectOne<EOT>
{
 public:
  /** Ctor
  */
  eoNoSelect(): current(0) {}

  virtual const EOT& operator()(const eoPop<EOT>& _pop)
  {
    if (current >= _pop.size())
      current=0;

    current++;
    return _pop[current-1] ;
  }
private:
  unsigned current;
};

#endif
