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

#include <iterator>
#include <list>

#include <EO.h>
#include <eoInit.h>
#include <eoOp.h>
#include <gp/parse_tree.h>

using namespace gp_parse_tree;

/** @defgroup ParseTree

Various functions for tree-based Genetic Programming

Example:
@include t-eoSymreg.cpp

@ingroup Representations
*/


/** Implementation of parse-tree for genetic programming

@class eoParseTree eoParseTree.h gp/eoParseTree.h

@ingroup ParseTree
*/
template <class FType, class Node>
class eoParseTree : public EO<FType>, public parse_tree<Node>
{
public:

    using parse_tree<Node>::back;
    using parse_tree<Node>::ebegin;
    using parse_tree<Node>::eend;
    using parse_tree<Node>::size;


    typedef typename parse_tree<Node>::subtree Subtree;

    /* For Compatibility with the intel C++ compiler for Linux 5.x */
    typedef Node reference;
    typedef const reference const_reference;

    /**
     * Default Constructor
     */
    eoParseTree(void)  {}

    /**
     * Copy Constructor
     * @param tree The tree to copy
     */
    eoParseTree(const parse_tree<Node>& tree)  : parse_tree<Node>(tree) {}

//    eoParseTree(const eoParseTree<FType, Node>& tree) :  parse_tree<Node>(tree) {}
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
            back() = this->operator[](size()-2);
        }
    }

    /**
     * To read me from a stream
     * @param is The std::istream
     */

    eoParseTree(std::istream& is) : EO<FType>(), parse_tree<Node>()
    {
        readFrom(is);
    }

    /// My class name
    std::string className(void) const { return "eoParseTree"; }

    /**
     * To print me on a stream
     * @param os The std::ostream
     */
    void printOn(std::ostream& os) const
    {
        EO<FType>::printOn(os);
        os << ' ';

        os << size() << ' ';

        std::copy(ebegin(), eend(), std::ostream_iterator<Node>(os, " "));
    }

    /**
     * To read me from a stream
     * @param is The std::istream
     */
    void readFrom(std::istream& is)
    {


        EO<FType>::readFrom(is);

        unsigned sz;
        is >> sz;


        std::vector<Node> v(sz);

        unsigned i;

        for (i = 0; i < sz; ++i)
        {
            Node node;
            is >> node;
            v[i] = node;
        }
        parse_tree<Node> tmp(v.begin(), v.end());
        this->swap(tmp);

        /*
         * old code which caused problems for paradisEO
         *
         * this can be removed once it has proved itself
        EO<FType>::readFrom(is);

        // even older code
        FType fit;
        is >> fit;

        fitness(fit);


        std::copy(std::istream_iterator<Node>(is), std::istream_iterator<Node>(), back_inserter(*this));
        */
    }
};
/** @example t-eoSymreg.cpp
 */

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
