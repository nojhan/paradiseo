/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    -----------------------------------------------------------------------------
    eoSequentialGOpSelector.h
      Sequential Generalized Operator Selector.

    (c) Maarten Keijzer (mkeijzer@mad.scientist.com), GeNeura Team 1998, 1999, 2000
 
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

#ifndef eoSequentialGOpSelector_h
#define eoSequentialGOpSelector_h

//-----------------------------------------------------------------------------

#include "eoGOpSelector.h"
/** eoSequentialGOpSel: do proportional selection, but return a sequence of
    operations to be applied one after the other. 
*/
template <class EOT> 
class eoSequentialGOpSel : public eoGOpSelector<EOT>
{
	public :
		
		virtual eoGeneralOp<EOT>& selectOp()
		{
            combined.clear();

			for (int i = 0; i < size(); ++i)
			{
				if (operator[](i) == 0)
					continue;

				if (rng.flip(getRates()[i]))
					combined.addOp(operator[](i));
			}

			return combined;
		}		

	private :

	eoCombinedOp<EOT> combined;
};

#endif
