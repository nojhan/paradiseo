/*	    
 *             Copyright (C) 2005 Maarten Keijzer
 *
 *          This program is free software; you can redistribute it and/or modify
 *          it under the terms of version 2 of the GNU General Public License as 
 *          published by the Free Software Foundation. 
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program; if not, write to the Free Software
 *          Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef NODESELECTOR_H
#define NODESELECTOR_H

#include <sym/Sym.h>

/** Base class for selecting nodes */
class NodeSelector {
    public:
	
    class NodeSelection {
	Sym root_;
	unsigned subtree_index_;
	Sym subtree_;

	public :
	    NodeSelection(Sym r, unsigned idx, Sym s) 
		: root_(r), subtree_index_(idx), subtree_(s) {}

	    Sym root() const { return root_; }
	    unsigned idx()  const { return subtree_index_; }
	    Sym subtree(); 
	
    };

    virtual ~NodeSelector() {}
	
    virtual NodeSelection select_node(Sym sym) const = 0;  
};


/** Select nodes uniformly */
class RandomNodeSelector : public NodeSelector {
    public:
    NodeSelection select_node(Sym sym) const;
};

/** A node selector that does a specified number of rounds ignoring terminals */
class BiasedNodeSelector : public NodeSelector {
    public:
    unsigned nRounds;

    BiasedNodeSelector() : nRounds(3) {} // 3: for binary trees 87.5% chance of selecting an internal node
    BiasedNodeSelector(unsigned n) : nRounds(n) {}
    
    NodeSelection select_node(Sym sym) const;
};

#endif
