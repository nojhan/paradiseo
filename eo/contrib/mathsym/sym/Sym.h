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

#ifndef SYMNODE_H_
#define SYMNODE_H_

#include <cassert>

#if __GNUC__ >= 3
#include <backward/hash_map.h>
#elif __GNUC__ < 3
#include <hash_map.h>
using std::hash_map;
#endif

/* Empty 'extra statistics' structure, derive from this to keep other characteristics of nodes */
struct UniqueNodeStats { virtual ~UniqueNodeStats(){} };

#include "SymImpl.h"
#include "token.h"

#if __GNUC__ == 4
#define USE_TR1 1
#else
#define USE_TR1 0
#endif
// TR1 is buggy at times
#undef USE_TR1
#define USE_TR1 0

#if USE_TR1
#include <tr1/unordered_map>
typedef std::tr1::unordered_map<detail::SymKey, detail::SymValue, detail::SymKey::Hash> SymMap;
#else
typedef hash_map<detail::SymKey, detail::SymValue, detail::SymKey::Hash> SymMap;
#endif

typedef SymMap::iterator SymIterator;

/* Sym is the tree, for which all the nodes are stored in a hash table. 
 * This makes checking for equality O(1) */
class Sym
{
    public:
	
	Sym() : node(dag.end()) {}
	explicit Sym(token_t token, const SymVec& args);
	explicit Sym(token_t token, const Sym& args);
	explicit Sym(token_t var);
	
	explicit Sym(SymIterator it) : node(it) { incref(); }
	
	Sym(const Sym& oth) : node(oth.node) { incref(); }
	~Sym() { decref(); }
	
	const Sym& operator=(const Sym& oth) {
	    if (oth.node == node) return *this;
	    decref();
	    node = oth.node;
	    incref();
	    return *this;
	}

	/* Unique Stats are user defined */
	UniqueNodeStats* extra_stats() const { return empty()? 0 : node->second.uniqueNodeStats; }
	
	int hashcode() const { return node->first.get_hash_code(); } //detail::SymKey::Hash hash; return hash(node->first); }
	
	// Friends, need to touch the node
	friend struct detail::SymKey::Hash;
	friend struct detail::SymKey;
	
	unsigned refcount() const { return empty()? 0: node->second.refcount; }

	bool operator==(const Sym& other) const {
	    return node == other.node;
	}
	bool operator!=(const Sym& other) const { return !(*this == other); }

	bool empty() const { return node == dag.end(); }

	/* Support for traversing trees */
	unsigned arity() const { return node->first.arity(); }
	token_t    token() const { return node->first.token; }
	
	const SymVec& args() const { return node->first.vec(); }
	
	/* size() - depth */
	unsigned size() const { return empty()? 0 : node->second.size; }
	unsigned depth() const { return empty()? 0 : node->second.depth; }
	
	SymMap::iterator iterator() const { return node; }

	/* Statics accessing some static members */
	static SymMap& get_dag() { return dag; }
	
	/* This function can be set to create some UniqueNodeStats derivative that can contain extra stats for a node,
	 * it can for instance be used to create ERC's and what not. */
	static void set_factory_function(UniqueNodeStats* (*f)(const Sym&)) { factory=f; } 
	static void clear_factory_function() { factory = 0; }
	
	static const std::vector<unsigned>& token_refcount() { return token_count; }
	
	unsigned address() const { return reinterpret_cast<unsigned>(&*node); }
	
    private :
	
	// implements getting subtrees
	Sym private_get(size_t w) const; 
	
	unsigned __unchecked_refcount() const { return node->second.refcount; }
	
	void incref() {
	    if (!empty()) {
		++(node->second.refcount);
		++token_count[token()];
	    }
	}
	void decref() {
	    if (!empty()) {
		--token_count[token()];
		if (--(node->second.refcount) == 0) {
		    dag.erase(node);
		}
	    }
	}

	// The one and only data member, an iterator into the static map below
	SymIterator node;
	
	// A static hash_map that contains all live nodes.. 
	static SymMap dag;
	
	static std::vector<unsigned> token_count;
	
	// Factory function for creating extra node stats, default will be 0
	static UniqueNodeStats* (*factory)(const Sym&);
	
};

/* Utility hash functor for syms */
class HashSym {
    public:
    int operator()(const Sym& sym) const { return sym.hashcode(); }
};

/* Utility Functions */

// get_subtree retrieves a subtree by standard ordering (0=root, and then depth first)
Sym get_subtree(const Sym& org, size_t w); 

// insert_subtree uses the same ordering as get and inserts the second argument, returning a new tree
Sym insert_subtree(const Sym& org, size_t w, const Sym& nw);

/* Get the successor from the hashtable, no particular purpose other than an interesting way to mutate */
inline Sym next(const Sym& sym) {
    SymIterator it = sym.iterator();
    ++it;
    if (it == Sym::get_dag().end()) it = Sym::get_dag().begin();
    return Sym(it);
}

#endif
