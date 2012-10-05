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

#ifndef EOSYMCROSSOVER_H
#define EOSYMCROSSOVER_H

class NodeSelector;
class Sym;

#include <eoOp.h>
    
extern bool subtree_quad(Sym& a, Sym& b, NodeSelector& select);
template <class EoType>
class eoQuadSubtreeCrossover : public eoQuadOp<EoType> {
    NodeSelector& node_selector;
    
    public:
    eoQuadSubtreeCrossover(NodeSelector& _node_selector) : node_selector(_node_selector) {}
    
    bool operator()(EoType& a, EoType& b) { return subtree_quad(a,b, node_selector); }
};


extern bool subtree_bin(Sym& a, const Sym& b, NodeSelector& select);
template <class EoType>
class eoBinSubtreeCrossover : public eoBinOp<EoType> {
    NodeSelector& node_selector;

    public :

    eoBinSubtreeCrossover(NodeSelector& _node_selector) : node_selector(_node_selector) {}

    bool operator()(EoType& a, const EoType& b) { return subtree_bin(a, b, node_selector); }
};

/** Yet another homologous crossover, afaik not particularly 
 * defined in the literature
 */
extern bool homologous_bin(Sym& a, const Sym& b);
template <class EoType>
class eoBinHomologousCrossover : public eoBinOp<EoType> {
    public:
	bool operator()(EoType& a, const EoType& b) {
	    return homologous_bin(a,b);
	}
};


extern bool size_level_xover(Sym& a, const Sym& b);
template <class EoType>
class eoSizeLevelCrossover : public eoBinOp<EoType> {
    public:
	bool operator()(EoType& a, const EoType& b) {
	    return size_level_xover(a,b);
	}
};

#endif

