/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoSelectPerc.h
   (c) Maarten Keijzer, GeNeura Team, 2000

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

#ifndef _eoSelectPerc_h
#define _eoSelectPerc_h


//-----------------------------------------------------------------------------
#include <eoSelect.h>
#include <eoSelectOne.h>
#include <math.h>
//-----------------------------------------------------------------------------

/** eoSelectPerc selects many individuals using eoSelectOne as it's
    mechanism. Therefore eoSelectPerc needs an eoSelectOne in its ctor

    It will select floor(rate*pop.size()) individuals and pushes them to
    the back of the destination population.

@ingroup Selectors
*/
template<class EOT>
class eoSelectPerc : public eoSelect<EOT>
{
 public:
     /// init
     eoSelectPerc(eoSelectOne<EOT>& _select, float _rate = 1.0)
         : select(_select), rate(_rate) {}

     /**
     The implementation selects a percentage

     @param _source the source population
     @param _dest  the resulting population (size of this population is the number of times eoSelectOne is called. It empties the destination and adds the selection into it)
     */
  virtual void operator()(const eoPop<EOT>& _source, eoPop<EOT>& _dest)
  {
    size_t target = static_cast<size_t>(floor(rate * _source.size()));

    _dest.resize(target);

    select.setup(_source);

    for (size_t i = 0; i < _dest.size(); ++i)
      _dest[i] = select(_source);
  }

private :
  eoSelectOne<EOT>& select;
  float rate;
};

#endif
