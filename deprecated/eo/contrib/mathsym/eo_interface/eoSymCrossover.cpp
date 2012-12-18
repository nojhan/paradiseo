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
#include <vector>

using namespace std;

bool subtree_quad(Sym& a, Sym& b, NodeSelector& select) {
    NodeSelector::NodeSelection sel_a = select.select_node(a);
    NodeSelector::NodeSelection sel_b = select.select_node(b);

    Sym aprime = insert_subtree(a, sel_a.idx(), sel_b.subtree() );
    Sym bprime = insert_subtree(b, sel_b.idx(), sel_a.subtree() );

    a = aprime;
    b = bprime;
    return true;
}

bool subtree_bin(Sym& a, const Sym& b, NodeSelector& select) {
    NodeSelector::NodeSelection sel_a = select.select_node(a);
    NodeSelector::NodeSelection sel_b = select.select_node(b);

    a = insert_subtree(a, sel_a.idx(), sel_b.subtree());

    return true;
}

Sym homologous_binimpl(Sym a, Sym b) {
    
    if(a == b) { return a; } // no point 
    
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

void set_size_levels(Sym sym, vector<unsigned>& l, vector<unsigned>& s, unsigned level = 1) {
    l.push_back(level);
    s.push_back(sym.size());
    
    for (unsigned i = 0; i < sym.args().size(); ++i) {
        set_size_levels(sym.args()[i], l, s, level+1);
    }
}


bool size_level_xover(Sym& a, const Sym& b) {
    
    Sym org = a;
        
    vector<unsigned> levela;
    vector<unsigned> sizesa;
    vector<unsigned> levelb;
    vector<unsigned> sizesb;

    set_size_levels(a, levela, sizesa);
    set_size_levels(b, levelb, sizesb);

    unsigned p0;
    unsigned p1;
    
    for (unsigned tries = 0;; ++tries) {
        p0 = rng.random(a.size());
        p1 = rng.random(b.size());
        
        if (tries < 5 && (sizesa[p0] != sizesb[p1] && levela[p0] != levelb[p1])) {
            continue;   
        }

        break;
    }

    a = insert_subtree(a, p0, get_subtree(b, p1));

    return org != a;
    
}


