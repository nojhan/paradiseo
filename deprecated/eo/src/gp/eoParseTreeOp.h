// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParseTreeOp.h : crossover and mutation operator for  the eoParseTree class
// (c) Maarten Keijzer 2000  for eoSubtreeXOver, eoBranchMutation
// (c) Jeroen Eggermont 2001 for other mutation operators

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

#ifndef eoParseTreeOp_h
#define eoParseTreeOp_h

#include <EO.h>
#include <eoOp.h>

#include <gp/eoParseTree.h>

/** eoSubtreeXOver --> subtree xover
\class eoSubtreeXOver eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
*/
template<class FType, class Node>
class eoSubtreeXOver: public eoQuadOp< eoParseTree<FType, Node> > {
public:

  typedef eoParseTree<FType,Node> EoType;
  /**
   * Constructor
   * @param _max_length the maximum size of an individual
   */
  eoSubtreeXOver( unsigned _max_length)
    : eoQuadOp<EoType>(), max_length(_max_length) {};

  /// the ckassname
  virtual std::string className() const { return "eoSubtreeXOver"; };

  /// Dtor
  virtual ~eoSubtreeXOver () {};

  /**
   * Perform crossover on two individuals
   * param _eo1 The first parent individual
   * param _eo2 The second parent individual
   */
  bool operator()(EoType & _eo1, EoType & _eo2 )
  {
          int i = rng.random(_eo1.size());
          int j = rng.random(_eo2.size());

          typename parse_tree<Node>::subtree tmp = _eo1[i];
          _eo1[i] = _eo2[j]; // insert subtree
          _eo2[j] = tmp;

          _eo1.pruneTree(max_length);
          _eo2.pruneTree(max_length);

    return true;
  }
 private:
  unsigned max_length;
};

/** eoBranchMutation --> replace a subtree with a randomly created subtree
\class eoBranchMutation eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
 */
template<class FType, class Node>
class eoBranchMutation: public eoMonOp< eoParseTree<FType, Node> >
{
public:

  typedef eoParseTree<FType,Node> EoType;
  /**
   * Constructor
   * @param _init An instantiation of eoGpDepthInitializer
   * @param _max_length the maximum size of an individual
   */
  eoBranchMutation(eoInit<EoType>& _init, unsigned _max_length)
    : eoMonOp<EoType>(), max_length(_max_length), initializer(_init)
  {};

  /// the class name
  virtual std::string className() const { return "eoBranchMutation"; };

  /// Dtor
  virtual ~eoBranchMutation() {};

  /**
   * Mutate an individual
   * @param _eo1 The individual that is to be changed
   */
  bool operator()(EoType& _eo1 )
  {
          int i = rng.random(_eo1.size());

      EoType eo2;
      initializer(eo2);

          int j = rng.random(eo2.size());

          _eo1[i] = eo2[j]; // insert subtree

          _eo1.pruneTree(max_length);

    return true;
  }

private :

  unsigned max_length;
  eoInit<EoType>& initializer;
};

// Additional Mutation operators from
// TITLE:"Genetic Programming~An Introduction"
// AUTHORS: Banzhaf, Nordin, Keller, Francone
// ISBN: 3-920993-58-6
// ISBN: 1-55860-510-X
//
// For the eoParseTree class

/** eoPointMutation --> replace a Node with a Node of the same arity
\class eoPointMutation eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
*/

template<class FType, class Node>
class eoPointMutation: public eoMonOp< eoParseTree<FType, Node> >
{
public:

  typedef eoParseTree<FType,Node> EoType;

  /**
   * Constructor
   * @param _initializor The std::vector of Nodes given to the eoGpDepthInitializer
   */
  eoPointMutation( std::vector<Node>& _initializor)
    : eoMonOp<EoType>(), initializor(_initializor)
  {};

  /// the class name
  virtual std::string className() const { return "eoPointMutation"; };

  /// Dtor
  virtual ~eoPointMutation() {};

  /**
   * Mutate an individual
   * @param _eo1 The individual that is to be changed
   */
  bool operator()(EoType& _eo1 )
  {
        // select a random node i that is to be mutated
        int i = rng.random(_eo1.size());
        // request the arity of the node that is to be replaced
        int arity = _eo1[i].arity();

        int j=0;

        do
        {
                j = rng.random(initializor.size());

        }while ((initializor[j].arity() != arity));

        _eo1[i] = initializor[j];



        return true;
  }

private :
        std::vector<Node>& initializor;

};

/** eoExpansionMutation --> replace a terminal with a randomly created subtree
\class eoExpansionMutation eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
 */

template<class FType, class Node>
class eoExpansionMutation: public eoMonOp< eoParseTree<FType, Node> >
{
public:

  typedef eoParseTree<FType, Node> EoType;

  /**
   * Constructor
   * @param _init An instantiation of eoGpDepthInitializer
   * @param _max_length the maximum size of an individual
   */
  eoExpansionMutation(eoInit<EoType>& _init, unsigned _max_length)
    : eoMonOp<EoType>(), max_length(_max_length), initializer(_init)
  {};

  /// The class name
  virtual std::string className() const { return "eoExpansionMutation"; };

  /// Dtor
  virtual ~eoExpansionMutation() {};
  /**
   * Mutate an individual
   * @param _eo1 The individual that is to be changed
   */
  bool operator()(EoType& _eo1 )
  {
          int i = rng.random(_eo1.size());
          // look for a terminal
          while (_eo1[i].arity() != 0)
          {
                i= rng.random(_eo1.size());
          };

          // create a new tree to
          EoType eo2;
          // make sure we get a tree with more than just a terminal
          do
          {
                initializer(eo2);
          }while(eo2.size() == 1);

          int j = rng.random(eo2.size());
          // make sure we select a subtree (and not a terminal)
          while((eo2[j].arity() == 0))
          {
                j = rng.random(eo2.size());
          };


          _eo1[i] = eo2[j]; // insert subtree

          _eo1.pruneTree(max_length);

    return true;
  }

private :

  unsigned max_length;
  eoInit<EoType>& initializer;
};

/** eoCollapseSubtree -->  replace a subtree with a randomly chosen terminal
\class eoCollapseSubtreeMutation eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
 */

template<class FType, class Node>
class eoCollapseSubtreeMutation: public eoMonOp< eoParseTree<FType, Node> >
{
public:

  typedef eoParseTree<FType,Node> EoType;
  /**
   * Constructor
   * @param _init An instantiation of eoGpDepthInitializer
   * @param _max_length the maximum size of an individual
   */
  eoCollapseSubtreeMutation(eoInit<EoType>& _init, unsigned _max_length)
    : eoMonOp<EoType>(), max_length(_max_length), initializer(_init)
  {};

  /// The class name
  virtual std::string className() const { return "eoCollapseSubtreeMutation"; };

  /// Dtor
  virtual ~eoCollapseSubtreeMutation() {};
  /**
   * Mutate an individual
   * @param _eo1 The individual that is to be changed
   */
  bool operator()(EoType& _eo1 )
  {
          int i = rng.random(_eo1.size());
          // look for a subtree
          while ((_eo1[i].arity() == 0) && (_eo1.size() > 1))
          {
                i= rng.random(_eo1.size());
          };

          // create a new tree to
          EoType eo2;
          initializer(eo2);

          int j = rng.random(eo2.size());
          // make sure we select a subtree (and not a terminal)
          while(eo2[j].arity() != 0)
          {
                j = rng.random(eo2.size());
          };

          _eo1[i] = eo2[j]; // insert subtree

          // we don't have to prune because the subtree is always smaller
          _eo1.pruneTree(max_length);

    return true;
  }

private :

  unsigned max_length;
  eoInit<EoType>& initializer;
};


/** eoHoistMutation -->  replace the individual with one of its subtree's
\class eoHoistMutation eoParseTreeOp.h gp/eoParseTreeOp.h
\ingroup ParseTree
 */

template<class FType, class Node>
class eoHoistMutation: public eoMonOp< eoParseTree<FType, Node> >
{
public:

  typedef eoParseTree<FType,Node> EoType;

  /**
   * Constructor
   */
  eoHoistMutation()
    : eoMonOp<EoType>()
  {};

  /// The class name
  virtual std::string className() const { return "eoHoistMutation"; };

  /// Dtor
  virtual ~eoHoistMutation() {};
  /**
   * Mutate an individual
   * @param _eo1 The individual that is to be changed
   */
  bool operator()(EoType& _eo1 )
  {


          // select a hoist point
          int i = rng.random(_eo1.size());
          // and create a new tree
          EoType eo2(_eo1[i]);

          // we don't have to prune because the new tree is always smaller
          //_eo1.pruneTree(max_length);

          _eo1 = eo2;

    return true;
  }

private :

};


#endif
