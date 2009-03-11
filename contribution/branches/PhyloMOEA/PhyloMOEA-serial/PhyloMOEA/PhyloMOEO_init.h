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

#ifndef PHYLOMOEO_INIT_H_
#define PHYLOMOEO_INIT_H_
#include <PhyloMOEO.h>
class Phyloraninit : public eoInit <PhyloMOEO>
{
	public:
		Phyloraninit(phylotreeIND &templatetree): tree(templatetree) { }

	void operator()(PhyloMOEO &ind)
	{
		ind.set_random_tree(tree);
	}	
	private:
		phylotreeIND &tree;
};

class Phylonewickinit : public eoInit <PhyloMOEO>
{
	public:
		Phylonewickinit(std::string newick): tree_string(newick) { }

	void operator()(PhyloMOEO &ind)
	{
		phylotreeIND &tree = ind.get_tree();
		tree.read_newick2( tree_string );
	}	
	private:
		std::string tree_string;
};
#endif

