/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoDetSelect.h
   (c) Marc Schoenauer, Maarten Keijzer, GeNeura Team, 2000

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
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoDetSelect_h
#define _eoDetSelect_h


//-----------------------------------------------------------------------------
#include <eoSelect.h>
#include <utils/eoHowMany.h>
#include <math.h>
//-----------------------------------------------------------------------------

/** eoDetSelect selects many individuals determinisctically
 *
 * @ingroup Selectors
*/
template<class EOT>
class eoDetSelect : public eoSelect<EOT>
{
 public:
     /// init
     eoDetSelect(double _rate = 1.0, bool _interpret_as_rate = true)
         : howMany(_rate, _interpret_as_rate)  {}

     /**
     @param _source the source population
     @param _dest  the resulting population (size of this population is
         given by the HowMany data
         It empties the destination and adds the selection into it)
     */
  virtual void operator()(const eoPop<EOT>& _source, eoPop<EOT>& _dest)
  {
    unsigned int pSize = _source.size();
    size_t target = howMany(pSize);

    if ( target == 0 )
        {
            eo::log << eo::warnings << "Call to a eoHowMany instance returns 0 (target=" << target << ") it will be replaced by 1 to continue." << std::endl;
            target = 1;
        }

    _dest.resize(target);

    unsigned remain = target % pSize;
    unsigned entireCopy = target / pSize;
    typename eoPop<EOT>::iterator it = _dest.begin();

    if (target >= pSize)
      {
        for (unsigned i=0; i<entireCopy; i++)
          {
              std::copy(_source.begin(), _source.end(), it);
            it += pSize;
          }
      }
    // the last ones
    if (remain)
      {
          std::copy(_source.begin(), _source.begin()+remain, it);
      }
  }

private :
  eoHowMany howMany;
};

#endif
