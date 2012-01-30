/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoBreed.h
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

#ifndef _eoBreed_h
#define _eoBreed_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoFunctor.h>
#include <eoSelect.h>
#include <eoTransform.h>
//-----------------------------------------------------------------------------

/** Breeding: combination of selecting and transforming a population

Breeding is thought of a combination of selecting and transforming a
population. For efficiency reasons you might want to build your own
eoBreed derived class rather than relying on a seperate select and
transform function.

@see eoSelect, eoTransform, eoSelectTransform

@ingroup Combination
*/
template<class EOT>
class eoBreed : public eoBF<const eoPop<EOT>&, eoPop<EOT>&, void>
{};



/** Embedded select, followed by an embedded transform

Special breeder that is just an application of an embedded select,
followed by an embedded transform

@ingroup Combination
*/
template <class EOT>
class eoSelectTransform : public eoBreed<EOT>
{
    public:
        eoSelectTransform(eoSelect<EOT>& _select, eoTransform<EOT>& _transform) :
        select(_select), transform(_transform)
        {}

        void operator()(const eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
        {
            select(_parents, _offspring);
            transform(_offspring);
        }

    private :
        eoSelect<EOT>& select;
        eoTransform<EOT>& transform;
};

#endif
