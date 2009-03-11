/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef VECTORSORTEDINDEX_H_
#define VECTORSORTEDINDEX_H_

#include <vector>
#include <algorithm>


template<typename T, typename C>  struct CmpObjIdx
{
		std::vector<T> &realvector;
		C &comp;

		CmpObjIdx(std::vector<T> &_realvector, C &_comp): realvector(_realvector), comp(_comp) {}
      	bool operator()(unsigned int a, unsigned int b)
        {
			return comp(realvector[a],realvector[b]);
        }
};

template< typename T, typename C > 
void vectorSortIndex( std::vector<T> &realvector, std::vector<unsigned int> &sortedvectoridx, C &comparator )
{
			CmpObjIdx<T,C> compidx(realvector, comparator);
			sortedvectoridx.resize( realvector.size());
			for(int i=0; i< realvector.size(); i++) sortedvectoridx[i] =i;
			std::sort( sortedvectoridx.begin(), sortedvectoridx.end(), compidx);
}
#endif


