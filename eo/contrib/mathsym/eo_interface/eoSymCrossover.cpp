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


#include <Sym.h>
#include <NodeSelector.h>

#include <eoSymCrossover.h>
#include <utils/eoRNG.h>

bool subtree_quad(Sym& a, Sym& b, NodeSelector& select) {
    unsigned i = select.select_node(a);
    unsigned j = select.select_node(b);

    Sym aprime = insert_subtree(a, i, get_subtree(b, j));
    Sym bprime = insert_subtree(b, j, get_subtree(a, i));

    a = aprime;
    b = bprime;
    return true;
}

bool subtree_bin(Sym& a, const Sym& b, NodeSelector& select) {
    unsigned i = select.select_node(a);
    unsigned j = select.select_node(b);

    a = insert_subtree(a, i, get_subtree(b,j));

    return true;
}

Sym homologous_binimpl(Sym a, Sym b) {

    bool use_a = rng.random(2);

    token_t head = (use_a? a : b).token();
    SymVec args = use_a?a.args() : b.args();

    const SymVec& a_args = a.args();
    const SymVec& b_args = b.args();
    unsigned mn = std::min(a_args.size(), b_args.size());
    
    bool changed = !use_a;
    
    for (unsigned i = 0; i < mn; ++i) {
	args[i] = homologous_binimpl(a_args[i], b_args[i]);
	if (args[i] != a_args[i]) {
	    changed = true;
	}
    }
    
    return changed? Sym(head, args) : a;
}

bool homologous_bin(Sym& a, const Sym& b) {
    if (a==b) return false;
    Sym org = a;
    a = homologous_binimpl(a,b);
    return org != a;
}


