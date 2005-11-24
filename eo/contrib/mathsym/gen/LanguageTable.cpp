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


#include "LanguageTable.h"
#include "Sym.h"

#include <utils/eoRNG.h>

using namespace std;

extern Sym default_const();

class LanguageImpl {
    public :
    std::vector<Sym> vars;
    LanguageTable::erc_func erc;
    
    std::vector<functor_t> functions;
    std::vector< std::vector<token_t> > functions_per_arity;

    LanguageImpl() : erc(default_const) {}
};

LanguageTable::LanguageTable() {
    pimpl = new LanguageImpl;
}

LanguageTable::~LanguageTable() {
    delete pimpl; 
}

LanguageTable::LanguageTable(const LanguageTable& that) {
    pimpl = new LanguageImpl(*that.pimpl);
}

LanguageTable& LanguageTable::operator=(const LanguageTable& that) {
    *pimpl = *that.pimpl;
    return *this;
}

void LanguageTable::add_function(token_t token, unsigned arity) {
    functor_t f = {token, arity};
    add_function( f );
}

void LanguageTable::add_function(functor_t f) {
     
    if (f.arity > 0) {
	pimpl->functions.push_back(f);
	
    } else {
	pimpl->vars.push_back(Sym(f.token));
    }
    
    if (pimpl->functions_per_arity.size() <= f.arity) pimpl->functions_per_arity.resize(f.arity+1);
    pimpl->functions_per_arity[f.arity].push_back(f.token);
    
}

void LanguageTable::set_erc( erc_func func) { pimpl->erc = func; }

/* Getting info out */

extern Sym SymConst(double val);

Sym LanguageTable::get_random_var()   const         { return rng.choice(pimpl->vars); }
Sym LanguageTable::get_random_const() const	    { return pimpl->erc(); }

functor_t LanguageTable::get_random_function() const 
{ 
    return rng.choice(pimpl->functions); 
}

token_t LanguageTable::get_random_function(token_t token, unsigned arity) const 
{ 
    if (pimpl->functions_per_arity.size() <= arity || pimpl->functions_per_arity[arity].empty()) {
	return token; // return original token if no functions of this arity are found
    }
    return rng.choice(pimpl->functions_per_arity[arity]); 
}



