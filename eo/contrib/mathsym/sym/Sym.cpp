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

#include <sstream>
#include <vector>

#include "Sym.h"

using namespace std;

typedef UniqueNodeStats* (*NodeStatFunc)(Sym&);

UniqueNodeStats* (*Sym::factory)(const Sym&) = 0;

SymMap Sym::dag(100000); // reserve space for so many nodes
std::vector<unsigned> Sym::token_count;
	

size_t get_size(const SymVec& vec) {
    size_t sz = 0;
    for (unsigned i = 0; i < vec.size(); ++i) {
	sz += vec[i].size();
    }
    return sz;
}

size_t get_depth(const SymVec& vec) {
    size_t dp = 1;
    for (unsigned i = 0; i < vec.size(); ++i) {
	dp = std::max(dp, vec[i].depth());
    }
    return dp;
}

Sym::Sym(token_t tok, const SymVec& args_) : node(dag.end())
{
    detail::SymKey key(tok, detail::SymArgs(args_));
    detail::SymValue val;

    node = dag.insert(pair<const detail::SymKey, detail::SymValue>(key, val)).first; 
    
    if (__unchecked_refcount() == 0) { // new node, set some stats 
	node->second.size  = 1 + get_size(args_);
	node->second.depth = 1 + get_depth(args_);
	
	// token count
	if (tok >= token_count.size()) {
	    token_count.resize(tok+1);
	}
	
	incref();
	node->first.fixate();	
	// call the factory function if available
	if (factory) node->second.uniqueNodeStats = factory(*this);
    
    }
    else incref();
}

Sym::Sym(token_t tok, const Sym& a) : node(dag.end()) { 
    SymVec args_; args_.push_back(a); 
    detail::SymKey key(tok, detail::SymArgs(args_));
    detail::SymValue val;

    node = dag.insert(pair<const detail::SymKey, detail::SymValue>(key, val)).first; 
    
    if (__unchecked_refcount() == 0) { // new node, set some stats 
	node->second.size = 1 + get_size(args_);
	node->second.depth = 1 + get_depth(args_);
	
	// token count
	if (tok >= token_count.size()) {
	    token_count.resize(tok+1);
	}
	
	incref();
	node->first.fixate();
	// call the factory function if available
	if (factory) node->second.uniqueNodeStats = factory(*this);
    }
    else incref();
}

Sym::Sym(token_t tok) : node(dag.end()) {
    detail::SymKey key(tok);
    detail::SymValue val;
    node = dag.insert(pair<const detail::SymKey, detail::SymValue>(key, val)).first; 
    
    if (__unchecked_refcount() == 0) { // new node, set some stats 
	node->second.size = 1;
	node->second.depth = 1;
	
	// token count
	if (tok >= token_count.size()) {
	    token_count.resize(tok+1);
	}
	
	incref();

	// call the factory function if available
	if (factory) node->second.uniqueNodeStats = factory(*this);
	
    }
    else incref();
}

std::pair<Sym,bool> insert_subtree_impl(const Sym& cur, size_t w, const Sym& nw) {
    if (w-- == 0) return make_pair(nw, !(nw == cur));
    
    const SymVec& vec = cur.args();
    std::pair<Sym,bool> result;
    unsigned i; 
    
    for (i = 0; i < vec.size(); ++i) {
	if (w < vec[i].size()) {
	    result = insert_subtree_impl(vec[i], w, nw);
	    if (result.second == false) return std::make_pair(cur, false); // unchanged
	    break;
	}
	w -= vec[i].size();
    }
    SymVec newvec = cur.args();
    newvec[i] = result.first;
    return make_pair(Sym(cur.token(), newvec), true);
}

Sym insert_subtree(const Sym& cur, size_t w, const Sym& nw) {
    return insert_subtree_impl(cur,w,nw).first;
}
Sym get_subtree(const Sym& cur, size_t w) {
    if (w-- == 0) return cur;
    
    const SymVec& vec = cur.args();
    for (unsigned i = 0; i < vec.size(); ++i) {
	if (w < vec[i].size()) return get_subtree(vec[i], w);
	w-=vec[i].size();
    }
    return cur;
}


