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
#include "Sym.h"

using namespace std;
namespace detail {
    
class SymArgsImpl {
    public:
    std::vector<Sym> owned_args;
};

size_t SymArgs::len() const { 
    return vec().size(); 
}

SymArgs::SymArgs() : pimpl( new SymArgsImpl ) {
    args_ptr = &pimpl->owned_args;
}

SymArgs::SymArgs(const std::vector<Sym>& v) : pimpl(0) {
    args_ptr = &v;
}

SymArgs::~SymArgs() { 
    delete pimpl; 
}

SymArgs::SymArgs(const SymArgs& args) : pimpl(0), args_ptr(args.args_ptr) {
    if (args.pimpl && args.args_ptr == &args.pimpl->owned_args) {
	pimpl = new SymArgsImpl(*args.pimpl);
	args_ptr = &pimpl->owned_args; 
    } 
}

SymArgs& SymArgs::operator=(const SymArgs& args) {
    if (args.pimpl && args.args_ptr == &args.pimpl->owned_args) {
	pimpl = new SymArgsImpl(*args.pimpl);
	args_ptr = &pimpl->owned_args;
    } else {
	args_ptr = args.args_ptr;
    }

    return *this;
}

void SymArgs::fixate() const {
    assert(pimpl == 0);
    pimpl = new SymArgsImpl;
    pimpl->owned_args = *args_ptr;
    args_ptr = &pimpl->owned_args;
}

// For Tackett's hashcode
#define PRIMET 21523
#define HASHMOD 277218551

const int nprimes = 4;
const unsigned long primes[] = {3221225473ul, 201326611ul, 1610612741ul, 805306457ul};
    
int SymKey::calc_hash() const {
    unsigned long hash = unsigned(token);
    hash *= PRIMET;
    
    const std::vector<Sym>& v = args.vec();
    for (unsigned i = 0; i < v.size(); ++i) {
	hash += ( (v[i].address() >> 3) * primes[i%nprimes]) % HASHMOD;
    }

    return hash;// % HASHMOD;
}

bool SymKey::operator==(const SymKey& other) const {
    if (token != other.token) return false;
    return args.vec() == other.args.vec();
}

/* Just to store this info somewhere:
 * 
 * Address Based Hash Function Implementation
 * uint32 address_hash(char* addr)
 * {
 *   register uint32 key;
 *     key = (uint32) addr;
 *       return (key >> 3) * 2654435761;
 *  }
 */     

SymValue::SymValue() : refcount(0), size(0), depth(0), uniqueNodeStats(0)  {}

SymValue::~SymValue() { delete uniqueNodeStats; }



} // namespace detail
