// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPopulator.h 
// (c) Maarten Keijzer and Marc Schoenauer, 2001
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

    Contact: mkeijzer@dhi.dk
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoPopulator_H
#define _eoPopulator_H

#include <eoPop.h>
#include <eoSelectOne.h>

/** eoPopulator is a helper class for general operators eoGenOp
    It is an eoPop but also behaves like an eoPop::iterator
    as far as operator* and operator++ are concerned

    See eoGenOp and eoOpContainer
*/
template <class EOT>
class eoPopulator : public eoPop<EOT>
{
public :

  eoPopulator(const eoPop<EOT>& _src) : current(begin()), src(_src) {}

  struct OutOfIndividuals {};

  /** a populator behaves like an iterator. Hence the operator*
   *      it returns the current individual -- eventually getting
   *      a new one through the operator++ if at the end
   */
  EOT& operator*(void)
  {
    if (current == end())
      operator++();

    return *current;
  }

  /** only prefix increment defined
   *  if needed, adds a new individual using the embedded selector
   *     and set the current pointer to the newly inserted individual
   *  otherwise simply increment the current pointer
   */
  eoPopulator& operator++()
  {
    if (current == end())
      { // get new individual from derived class select()
        push_back(select());
        current = end();
        --current;
        return *this;
      }
    // else
    ++current;
    return *this;
  }

  /** mandatory for operators that generate more offspring than parents
   *  if such a thing exists ?
   */
  void insert(const EOT& _eo)
  { /* not really efficient, but its nice to have */
    current = eoPop<EOT>::insert(current, _eo);
  }

  /** just to make memory mangement more efficient
   */
  void reserve(int how_many)
  {
    size_t sz = current - begin();
    eoPop<EOT>::reserve(size() + how_many);
    current = begin() + sz;
  }

  /** can be useful for operators with embedded selectors
   *  e.g. your barin and my beauty -type
   */
  const eoPop<EOT>& source(void) { return src; }

  typedef unsigned position_type;

  /** this is a direct access container: tell position */
  position_type tellp()         { return current - begin(); }
  /** this is a direct access container: go to position */
  void seekp(position_type pos) { current = begin() + pos; }
  /** no more individuals  */
  bool exhausted(void)          { return current == end(); }

  virtual const EOT& select() = 0;

protected :
  /** the pure virtual selection method - will be instanciated in
   *   eoSeqPopulator and eoPropPopulator
   */
  eoPop<EOT>::iterator current;
  const eoPop<EOT>& src;
};


/** SeqPopulator: an eoPopulator that sequentially goes through the population
is supposed to be used after a batch select of a whole bunch or genitors
 */
template <class EOT>
class eoSeqPopulator : public eoPopulator<EOT>
{
public :

  eoSeqPopulator(const eoPop<EOT>& _pop) :
    eoPopulator<EOT>(_pop), src_it(_pop.begin()) {}

  const EOT& select(void)
  {
    if (src_it == src.end())
      {
	throw OutOfIndividuals();
      }

    const EOT& res = *src_it++;
    return res;
  }

private :
  vector<EOT>::const_iterator src_it;
};


/** SelectivePopulator an eoPoplator that uses an eoSelectOne to select guys.
Supposedly, it is passed the initial population.
 */
template <class EOT>
class eoSelectivePopulator : public eoPopulator<EOT>
{
public :
  eoSelectivePopulator(const eoPop<EOT>& _pop, eoSelectOne<EOT>& _sel)
    : eoPopulator<EOT>(_pop), sel(_sel) {}

  const EOT& select()
  {
    return sel(src);
  }

private :
  eoSelectOne<EOT>& sel;
};

#endif

