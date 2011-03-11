/***************************************************************************
 *   Copyright (C) 2005 by Waldo Cancino                                   *
 *   wcancino@icmc.usp.br                                                  *
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

#ifndef _treeIterator_H_
#define _treeIterator_H_

#include <list>
#include <GTL/graph.h>


typedef pair<node, node::inout_edges_iterator> stack_info;
// Virtual Base Class for Pre-Order and Post-Order iterators
class treeIterator
{
	protected:
		node invalid_node; // points to the virtual root
		node root;		// root of the tree (subtree)
		node father;		// father of the root (tree or subtree)
		node curnode;		// current node
		node curfather;	// father of the current node
		// stack saves the father nodes and the iterator position for the
		// current node (child of father)
		node::inout_edges_iterator current_it;
	
		list< stack_info > pilha;
	public:
		// constructor for an sub-tree pre-order traversal
		treeIterator(node r, node f): root(r), father(f) {};
		// constructor for a tree pre-order traversal father = invalid_node
		treeIterator(node r): root(r), father(invalid_node) {};
		virtual ~treeIterator() {};
		// compare two iterators
		inline bool operator!=(treeIterator &other)
		{	
			return curnode != other.curnode;
		}
		inline void last_node() { pilha.clear(); curnode = curfather = father; }
		// pointer operator
		inline node * operator->() { return &curnode; }
		// dereference operator
		inline node & operator*() { return curnode; }
		// ancestor of the current_node
		inline node ancestor() { return curfather; }
		// branch with endpoints (curnode, curfather)
		inline edge branch() { return *(pilha.back().second); }
		// size of the stack
		inline int stack_size() { return pilha.size(); }
		// point to the first child of curnode
		node::inout_edges_iterator& first_valid_edge();
		// point to the next child of curnode
		node::inout_edges_iterator& next_valid_edge();
		// defined in derived classes
		virtual void first_node() = 0;
};

class child_Iterator: public treeIterator
{
	public:
		child_Iterator(node r, node f):treeIterator(r,f) {};
		child_Iterator(node r):treeIterator(r) {};
		child_Iterator& operator++();
		void first_node();
		edge branch() { return *current_it; };
		virtual ~child_Iterator() {};
};

class postorder_Iterator: public treeIterator
{
	public:
		postorder_Iterator(node r, node f):treeIterator(r,f) {};
		postorder_Iterator(node r):treeIterator(r) {};
		postorder_Iterator& operator++();
		void first_node();
		virtual ~postorder_Iterator() {};
};

class preorder_Iterator: public treeIterator
{
	public:
		preorder_Iterator(node r, node f): treeIterator(r,f) {};
		preorder_Iterator(node r):treeIterator(r)  {};
		preorder_Iterator& operator++();
		void first_node();
		virtual ~preorder_Iterator() {};
};

#endif
