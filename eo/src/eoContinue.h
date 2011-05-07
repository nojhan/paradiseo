// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoContinue.h
// (c) Maarten Keijzer, Geneura Team, 1999, 2000
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
 */
//-----------------------------------------------------------------------------

#ifndef _eoContinue_h
#define _eoContinue_h

#include <eoFunctor.h>
#include <eoPop.h>
#include <eoPersistent.h>

/** @defgroup Continuators Stopping criteria
 *
 * A stopping criterion is called a "continue". This is a functor that is called at each generation end
 * and that return true if one should stop the search.
 *
 *  @ingroup Utilities
 */

/** Termination condition for the genetic algorithm
 * Takes the population as input, returns true for continue,
 * false for termination
 *
 * @ingroup Continuators
 * @ingroup Core
 */
template< class EOT>
class eoContinue : public eoUF<const eoPop<EOT>&, bool>, public eoPersistent
{
public:
  virtual std::string className(void) const { return "eoContinue"; }

  /** Read from a stream
   * @param __is istream to read from
   */
  void readFrom (std :: istream & __is) {
      (void)__is;
    /* It should be implemented by subclasses ! */
  }

  /** Print on a stream
   * @param __os ostream to print on
   */
  void printOn (std :: ostream & __os) const {
      (void)__os;
    /* It should be implemented by subclasses ! */
  }
};

#endif
