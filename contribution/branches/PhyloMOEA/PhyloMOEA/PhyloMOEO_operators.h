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
#ifndef _PHYLOMOEO_OPERATORS_H_
#define _PHYLOMOEO_OPERATORS_H_
#include <PhyloMOEO.h>


class Phylocross : public eoQuadOp < PhyloMOEO >
  {
  public:

    /**
     * the class name (used to display statistics)
     */
    std::string className() const { return "Phylocross"; } ;


    bool operator()(PhyloMOEO & _tree1, PhyloMOEO & _tree2)
	{
		phylotreeIND parent_1(_tree1.get_tree());
		phylotreeIND parent_2(_tree2.get_tree());
		phylotreeIND &son1 = _tree1.get_tree();
		phylotreeIND &son2 = _tree2.get_tree();
		parent_1.export_subtree( son2 );
		parent_2.export_subtree( son1 );
		parent_1.invalidate_splits();
		parent_2.invalidate_splits();
		return true;
	}
  };

class Phylomutate : public eoMonOp <PhyloMOEO>
{
	public:
	std::string className() const { return "Phylomut"; } 

    bool operator() (PhyloMOEO & _tree)
	{
		_tree.get_tree().mutate(1);
		return true;
	}

};
#endif
