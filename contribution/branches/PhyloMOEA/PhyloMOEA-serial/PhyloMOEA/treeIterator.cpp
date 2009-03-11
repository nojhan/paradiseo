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

#include "treeIterator.h"



node::inout_edges_iterator& treeIterator::first_valid_edge()
{
	assert(curnode != curfather);
	current_it = curnode.inout_edges_begin();
	node tmp = current_it->opposite(curnode);
	// a valid edge from current_node can't point to the father
	if(tmp == curfather) current_it++;
	return current_it;
}

node::inout_edges_iterator& treeIterator::next_valid_edge()
{
	assert(curnode != curfather);
	current_it++;
	if( current_it == curnode.inout_edges_end() ) return current_it;
	node tmp = current_it->opposite(curnode);
	// a valid edge from current_node can't point to the father
	if( tmp == curfather ) current_it++;
	return current_it;
}

void child_Iterator::first_node()
{
	curnode = root;
	curfather = father;
	current_it = first_valid_edge();
	if(current_it != root.inout_edges_end())
		curnode = current_it->opposite(curnode);
}

child_Iterator& child_Iterator::operator++()
{
	if( curnode == root) curnode = father;
	else
	{
		curnode = root;
		current_it = next_valid_edge();
		// end 
		if(current_it == root.inout_edges_end())
			curnode = father;
		else
			curnode = current_it->opposite(curnode);
	}
	return (*this);
}


void postorder_Iterator::first_node()
{
	curnode = root;
	curfather = father;
	node::inout_edges_iterator it;
	pilha.clear();
	pilha.push_back( stack_info (father,it) );
	// seek for the first terminal node
	while( (current_it = first_valid_edge())  != curnode.inout_edges_end() )
	{
		pilha.push_back( stack_info (curnode, current_it) );
		//cout << curnode <<  " (" << curnode.degree() << ")" << ",";
		curfather = curnode;
		curnode = current_it->opposite(curnode);
	}		
}

postorder_Iterator& postorder_Iterator::operator++()
{
	// final node curnode == root --> curnode_next = end
	if( curnode == root) curnode = father;
	else
	{
		// current father becomes curent node
		// and grapndfather becomes father
		curnode = curfather;
		current_it = pilha.back().second;
		curfather = (++pilha.rbegin())->first;
		// get next children
		//current_it = next_valid_edge();
		//;
		
		if( next_valid_edge() != curnode.inout_edges_end() )
		{
			// update current node iterator
			
			pilha.back().second = current_it; 
			curfather = curnode;
			curnode = current_it->opposite(curnode);
			// get the next post-order node
			while( first_valid_edge() != curnode.inout_edges_end() )
			{
				pilha.push_back( stack_info (curnode, current_it) );
				curfather = curnode;
				curnode = current_it->opposite(curnode);
			}		
		}
		else
		{
			// pop the stack...  update curnode and curfather
			//curnode = pilha.back().first;
			//current_it = pilha.back().second;
			pilha.pop_back();
			//curfather = pilha.back().first;
		}
	}
	return (*this);
}

void preorder_Iterator::first_node()
{
	// the first node is root, clean the stack
	pilha.clear();
	node::inout_edges_iterator it;
	pilha.push_back( stack_info (father,it) );
	curnode = root;  curfather = father;
}

preorder_Iterator& preorder_Iterator::operator++()
{
	if( first_valid_edge() != curnode.inout_edges_end() )
	{
		pilha.push_back( stack_info (curnode, current_it) );
		curfather = curnode;
		curnode = current_it->opposite(curnode);
		//cout << "proximo no: " << curnode << endl;
	}
	else
	{
		curnode = curfather;
		current_it = pilha.back().second;
		curfather = (++pilha.rbegin())->first;
		// pop all nodes which all childs has been already visited
		while( curnode != father && next_valid_edge() == curnode.inout_edges_end() )
		{
			
			//cout << " eliminando.." << curnode;
			
			pilha.pop_back();
			curnode = pilha.back().first;
			current_it = pilha.back().second;
			if(curnode !=father)curfather = (++pilha.rbegin())->first;
		}
		// final node reached
		if( curnode == father){
				// curnode = father;
					cout << "putz cheguei ao final...!" << endl;
		}
		else
		{
			curfather = curnode;
			curnode = current_it->opposite(curnode);
			pilha.back().second = current_it;
			//cout << "pai.." << curfather << "filho..." << curnode << "   !";
		}
	}
	return (*this);
}
