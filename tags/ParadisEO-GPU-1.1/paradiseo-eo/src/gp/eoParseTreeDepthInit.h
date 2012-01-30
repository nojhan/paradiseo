// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParseTreeDepthInit.h : initializor for eoParseTree class
// (c) Maarten Keijzer 2000  Jeroen Eggermont 2002
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
             jeggermo@liacs.nl

 */
//-----------------------------------------------------------------------------

#ifndef eoParseTreeDepthInit_h
#define eoParseTreeDepthInit_h

#include <EO.h>
#include <gp/eoParseTree.h>
#include <eoInit.h>
#include <eoOp.h>
#include <eoPop.h>

using namespace gp_parse_tree;

/** eoParseTreeDepthInit : the initializer class for eoParseTree
\class eoParseTreeDepthInit eoParseTreeDepthInit.h gp/eoParseTreeDepthInit.h
\ingroup ParseTree
*/

// eoGpDepthInitializer is defined for backward compatibility
#define eoGpDepthInitializer eoParseTreeDepthInit

template <class FType, class Node>
class eoParseTreeDepthInit : public eoInit< eoParseTree<FType, Node> >
{
    protected:
        // a binary predicate for sorting
        // hopefully this will work with M$VC++ 6.0
        struct lt_arity:public std::binary_function<Node,Node,bool>
        {
                bool operator()(const Node &_node1, const Node &_node2) { return (_node1.arity() < _node2.arity());};
        };

    public :

    typedef eoParseTree<FType, Node> EoType;

    /**
     * Constructor
     * @param _max_depth The maximum depth of a tree
     * @param _initializor A std::vector containing the possible nodes
     * @param _grow False results in a full tree, True result is a randomly grown tree
     * @param _ramped_half_and_half True results in Ramped Half and Half Initialization
     */
        eoParseTreeDepthInit(
        unsigned _max_depth,
        const std::vector<Node>& _initializor,
        bool _grow = true,
        bool _ramped_half_and_half = false)
            :
            eoInit<EoType>(),
                        max_depth(_max_depth),
                        initializor(_initializor),
                        grow(_grow),
                        ramped_half_and_half(_ramped_half_and_half),
                        current_depth(_max_depth)
    {
      if(initializor.empty())
      {
        throw std::logic_error("eoParseTreeDepthInit: uhm, wouldn't you rather give a non-empty set of Nodes?");
      }
      // lets sort the initializor std::vector according to  arity (so we can be sure the terminals are in front)
      // we use stable_sort so that if element i was in front of element j and they have the same arity i remains in front of j
      stable_sort(initializor.begin(), initializor.end(), lt_arity());
    }
        /// My class name
        virtual std::string className() const { return "eoParseTreeDepthInit"; };

    /**initialize a tree
     * @param _tree : the tree to be initialized
     */
    void operator()(EoType& _tree)
        {
        std::list<Node> sequence;
        generate(sequence, current_depth);

        parse_tree<Node> tmp(sequence.begin(), sequence.end());
        _tree.swap(tmp);

        if(ramped_half_and_half)
        {
                if(grow)
                {
                        if (current_depth > 2)
                                current_depth--;
                        else
                                current_depth = max_depth;
                }
                // change the grow method from 'grow' to 'full' or from 'full' to 'grow'
                grow = !grow;
        };

        }
   private :
    void generate(std::list<Node>& sequence, int the_max, int last_terminal = -1)
    {
            if (last_terminal == -1)
            { // check where the last terminal in the sequence resides
            typename std::vector<Node>::iterator it;
                    for (it = initializor.begin(); it != initializor.end(); ++it)
                    {
                            if (it->arity() > 0)
                                    break;
                    }

                    last_terminal = it - initializor.begin();
            }

            if (the_max == 1)
            { // generate terminals only
                    typename std::vector<Node>::iterator it = initializor.begin() + rng.random(last_terminal);
                    it->randomize();
                    sequence.push_front(*it);
                    return;
            }

            typename std::vector<Node>::iterator what_it;

            if (grow)
            {
                    what_it = initializor.begin() + rng.random(initializor.size());
            }
            else // full
            {
                    what_it = initializor.begin() + last_terminal + rng.random(initializor.size() - last_terminal);
            }

        what_it->randomize();

            sequence.push_front(*what_it);

            for (int i = 0; i < what_it->arity(); ++i)
                    generate(sequence, the_max - 1, last_terminal);
    }



        unsigned max_depth;
        std::vector<Node> initializor;
        bool grow;
        bool ramped_half_and_half;
        unsigned current_depth;
};

/**
     * A template function for ramped half and half initialization of an eoParseTree population
     * @param pop the population to be created
     * @param population_size the size of the population to be created
     * @param init_max_depth the initial maximum tree depth
     * @param initializor A std::vector containing the possible nodes

     \ingroup ParseTree
     */
template <class FType, class Node>
void  eoInitRampedHalfAndHalf(eoPop< eoParseTree<FType,Node> > &pop, unsigned int population_size, unsigned int init_max_depth, std::vector<Node> &initializor)
{
        typedef eoParseTree<FType,Node> EoType;
        typedef eoPop< EoType > Pop;

        unsigned int M = init_max_depth - 1;
        unsigned int part_pop_size = population_size / (2*M);
        unsigned int m=0;

        std::cerr << "EO WARNING: Ramped Half and Half Initialization is now supported by eoParseTreeDepthInit." << std::endl;
        std::cerr << "            This function is now obsolete and might be removed in the future so you should"<< std::endl;
        std::cerr << "            update your code to use: " << std::endl << std::endl;
        std::cerr << "            eoParseTreeDepthInit( _max_depth, _initializer, bool _grow, bool _ramped_half_and_half)" << std::endl << std::endl;

        pop.clear();

        // initialize with Depth's (D) -> 2
        for(m=init_max_depth; m >= 2; m--)
        {
                eoParseTreeDepthInit<FType, Node> grow_initializer(m, initializor, true);
                Pop grow(part_pop_size, grow_initializer);
                pop.insert(pop.begin(), grow.begin(), grow.end());

                eoParseTreeDepthInit<FType, Node> full_initializer(m, initializor, false);
                Pop full(part_pop_size, full_initializer);
                pop.insert(pop.begin(), full.begin(), full.end());
        }

        bool g = true;
        while (pop.size() < population_size)
        {
                eoParseTreeDepthInit<FType, Node> initializer(init_max_depth, initializor, g);
                Pop p(1, initializer);
                pop.insert(pop.begin(), p.begin(), p.end());
                g= !g;
        }
}


#endif
