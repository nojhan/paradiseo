// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
 
//-----------------------------------------------------------------------------
// eoParseTree.h : eoParseTree class (for Tree-based Genetic Programming)
// (c) Maarten Keijzer 2000  
/*
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
    	     mak@dhi.dk 
            
 */
//-----------------------------------------------------------------------------

#ifndef eoParseTree_h
#define eoParseTree_h

#include <list>

#include <EO.h>
#include <eoOp.h>
#include <gp/parse_tree.h>
#include <eoInit.h>

using namespace gp_parse_tree;
using namespace std;

/**
\defgroup ParseTree

  Various functions for tree-based Genetic Programming
*/

/** eoParseTree : implementation of parse-tree for genetic programming
\class eoParseTree eoParseTree.h gp/eoParseTree.h
\ingroup ParseTree
*/


template <class FType, class Node>
class eoParseTree : public EO<FType>, public parse_tree<Node>
{
public :

    typedef parse_tree<Node>::subtree Subtree;

   /* For Compatibility with the intel C++ compiler for Linux 5.x */
   typedef Node reference;
   typedef const reference const_reference;
    
    

	
    /**
     * Default Constructor
     */
    eoParseTree(void) : EO<FType>() {}
    /** 
     * Copy Constructor
     * @param tree The tree to copy
     */
    eoParseTree(const parse_tree<Node>& tree) : EO<FType>(), parse_tree<Node>(tree) {}
    
    eoParseTree(const eoParseTree<FType, Node>& tree) : EO<FType>(), parse_tree<Node>(tree) {}
    /**
     * To prune me to a certain size
     * @param _size My maximum size
     */
    virtual void pruneTree(unsigned _size)
    {
        if (_size < 1)
            return;

        while (size() > _size)
        {
            back() = operator[](size()-2); 
        }
    }

    /**
     * To read me from a stream
     * @param is The istream
     */
     
    eoParseTree(std::istream& is) : EO<FType>(), parse_tree<Node>() 
    {
        readFrom(is);
    }

    /// My class name
    string className(void) const { return "eoParseTree"; }

    /**
     * To print me on a stream
     * @param os The ostream
     */
    void printOn(std::ostream& os) const
    {
	
	EO<FType>::printOn(os);
	/*
	 * old code which caused problems for paradisEO
	 * now we use EO<FType>::readFrom(is)
	 *
        os << fitness() << ' ';
	*/
        std::copy(ebegin(), eend(), ostream_iterator<Node>(os));
    }
    
    /**
     * To read me from a stream
     * @param is The istream
     */
    void readFrom(std::istream& is) 
    {
        EO<FType>::readFrom(is);
	
	
	/*
	 * old code which caused problems for paradisEO
	 * now we use EO<FType>::readFrom(is)
	 *
	FType fit;
        is >> fit;

        fitness(fit);
	*/

        std::copy(istream_iterator<Node>(is), istream_iterator<Node>(), back_inserter(*this));
    }
};

// friend function to print eoParseTree
template <class FType, class Node>
std::ostream& operator<<(std::ostream& os, const eoParseTree<FType, Node>& eot)
{
    eot.printOn(os);
    return os;
}

// friend function to read eoParseTree
template <class FType, class Node>
std::istream& operator>>(std::istream& is, eoParseTree<FType, Node>& eot)
{
    eot.readFrom(is);
    return is;
}

// for backward compatibility
#include <gp/eoParseTreeOp.h>
#include <gp/eoParseTreeDepthInit.h>

#endif
