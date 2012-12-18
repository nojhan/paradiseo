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
#ifndef __SYM_IMPL_H__
#define __SYM_IMPL_H__

#include <vector>

#include "token.h"

class Sym;

#if __GNUC__ > 4 
#include <ext/pool_allocator.h>
typedef std::vector<Sym, __gnu_cxx::__pool_alloc<Sym> > std::vector<Sym>;
//typedef std::vector<Sym> SymVec;
#else
typedef std::vector<Sym> SymVec;
#endif


namespace detail {

class SymArgsImpl;
class SymArgs {
    
    mutable SymArgsImpl* pimpl; // contains circular reference to vector<Sym>
    mutable const std::vector<Sym>* args_ptr;
    
    public:

    SymArgs();
    SymArgs(const std::vector<Sym>& v);
    ~SymArgs();

    SymArgs(const SymArgs& args);
    SymArgs& operator=(const SymArgs& other);

    size_t len() const;
    const std::vector<Sym>& vec() const { return *args_ptr; }
    void fixate() const;
};

class SymKey
{
    public:
	SymKey(token_t _token) : args(), token(_token), hash_code(calc_hash()) {}
	SymKey(token_t _token, const detail::SymArgs& _args) : args(_args), token(_token), hash_code(calc_hash()) {}
	
	bool operator==(const SymKey& other) const;
	
	struct Hash
	{
	    int operator()(const SymKey& k) const { return k.calc_hash(); }; 
	};
	
	unsigned arity() const { return args.len(); }
	const std::vector<Sym>& vec() const { return args.vec(); }
	
	// fixates (i.e. claims memory) for the embedded vector of Syms
	void fixate() const { args.fixate(); }
	
	int get_hash_code() const { return hash_code; }
	
	detail::SymArgs args;
	token_t token;     // identifies the function

    private:
	int calc_hash() const;
	int hash_code;
};

struct SymValue
{
    friend class Sym;
     
    SymValue();
    ~SymValue(); 
    
    unsigned getRefCount() const { return refcount; }
    unsigned getSize() const     { return size; }
    unsigned getDepth() const    { return depth; }
    
     
    
    // for reference counting
    unsigned refcount;
    
    // some simple stats
    unsigned size;
    unsigned depth;
    UniqueNodeStats* uniqueNodeStats;
    
};


} // namespace detail

#endif

