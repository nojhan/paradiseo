// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStParseTreeDepthInit.h : initializor strongly type GP
// (c) Jeroen Eggermont 2001
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
             jeggermo@liacs.nl

 */
//-----------------------------------------------------------------------------

#ifndef eoStParseTreeDepthInit_h
#define eoStParseTreeDepthInit_h

#include <EO.h>
#include <gp/eoParseTree.h>
#include <eoInit.h>
#include <eoOp.h>

#include <map>

using namespace gp_parse_tree;

#define TERMINAL 0

#define NONTERMINAL 4
#define ALL 5

 /**
\defgroup StParseTree

  Various functions for strongly typed tree-based Genetic Programming.
  The StParseTree functions use the same eoParseTree class for the
  individual but now each node class must have two additional functions.
  \li int type(void) which returns the return type of the node
  \li int type(int child) which returns the required type for child 0, 1 or 2

  Pruning strongly typed trees is not possible at the moment.

  \ingroup Representations
*/

/** eoStParseTreeDepthInit : the initializer class for strongly typed tree-based genetic programming
\class eoStParseTreeDepthInit eoStParseTreeDepthInit.h gp/eoStParseTreeDepthInit.h
\ingroup StParseTree
*/

template <class FType, class Node>
class eoStParseTreeDepthInit : public eoInit< eoParseTree<FType, Node> >
{
    public :

    typedef eoParseTree<FType, Node> EoType;

    /**
     * Constructor
     * @param _max_depth The maximum depth of a tree
     * @param _node A std::vector containing the possible nodes
     * @param _return_type (JD_2010-11-09: don't know the use of this parameter, maybe to force implicit template instanciation?)
     * @param _grow False results in a full tree, True result is a randomly grown tree
     */
        eoStParseTreeDepthInit(
        unsigned _max_depth,
                const std::vector<Node>& _node,
                const int& _return_type,
        bool _grow = true)
            :
            eoInit<EoType>(),
                        max_depth(_max_depth),
                        return_type(_return_type),
                        grow(_grow)
    {
      if(_node.empty())
      {
        throw std::logic_error("eoStParseTreeDepthInit: uhm, wouldn't you rather give a non-empty set of Nodes?");
      }


      unsigned int i=0;
      int arity=0;
      int type=0;
      std::vector<Node> node_vector;
      for(i=0; i < _node.size(); i++)
      {
        arity = _node[i].arity();
        type = _node[i].type();
        if(arity==0)
        {
                node_vector = node[type][TERMINAL];
                node_vector.push_back(_node[i]);
                node[type][TERMINAL]= node_vector;
        }
        else
        //if (arity != 0) // non-terminal
        {
                node_vector = node[type][NONTERMINAL];
                node_vector.push_back(_node[i]);
                node[type][NONTERMINAL] = node_vector;
        }
        node_vector = node[type][ALL];
        node_vector.push_back(_node[i]);
        node[type][ALL] = node_vector;

      }


    }
        /// My class name
        virtual std::string className() const { return "eoStParseTreeDepthInit"; };

    /**initialize a tree
     * @param _tree : the tree to be initialized
     */
    void operator()(EoType& _tree)
        {
                std::list<Node> sequence;
                bool good_tree = false;
                do
                {
                        sequence.clear();
                        good_tree = generate(sequence, max_depth, return_type);
                }while (!good_tree);

                parse_tree<Node> tmp(sequence.begin(), sequence.end());
                _tree.swap(tmp);
        }
   private :
    bool generate(std::list<Node>& sequence, int the_max, int request_type)
    {

            int selected=0;
            bool ok = true;

            if (the_max == 1)
            { // generate terminals only
                if( node[request_type][TERMINAL].empty() ) // no possible terminal node of this type
                        return false; // we have an invalid tree
                else
                {
                        selected = rng.random((node[request_type][TERMINAL]).size());
                        sequence.push_front(node[request_type][TERMINAL][selected]);
                        return true;
                }

            }

            int arity=0;
            if (grow)
            {
                selected = rng.random((node[request_type][ALL]).size());
                arity = node[request_type][ALL][selected].arity();
                sequence.push_front(node[request_type][ALL][selected]);
                for (int i = 0; i < arity; ++i)
                    ok &= generate(sequence, the_max - 1, node[request_type][ALL][selected].type(i));
            }
            else // full
            {
                 selected = rng.random((node[request_type][NONTERMINAL]).size());
                 arity = node[request_type][NONTERMINAL][selected].arity();
                 sequence.push_front(node[request_type][NONTERMINAL][selected]);
                 for (int i = 0; i < arity; ++i)
                     ok &=generate(sequence, the_max - 1, node[request_type][NONTERMINAL][selected].type(i));
            }

            return ok;

    }




        unsigned max_depth;
        std::map < int, std::map < int, std::vector<Node> > > node;

        int return_type;
        bool grow;
};

#endif
