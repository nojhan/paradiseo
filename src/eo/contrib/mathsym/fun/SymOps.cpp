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

#include "FunDef.h"
#include "SymOps.h"
#include "Sym.h"

using namespace std;

void collect(token_t t, Sym a, SymVec& args) {
    
    if (a.token() == t) {
	const SymVec& a_args = a.args();
	for (unsigned i = 0; i < a_args.size(); ++i) {
	    collect(t, a_args[i], args);
	}
	return;
    }
     
    args.push_back(a);
}

Sym operator+(Sym a, Sym b) {
    
    SymVec args;

    collect(sum_token, a, args);
    collect(sum_token, b, args);
    
    return Sym(sum_token, args);
}

Sym operator*(Sym a, Sym b) {
    
    SymVec args;

    collect(prod_token, a, args);
    collect(prod_token, b, args);
    
    return Sym(prod_token, args);
}

Sym operator/(Sym a, Sym b) {
    
    SymVec args;

    collect(prod_token, a, args);
    
    SymVec args2;
    collect(prod_token, b, args2);
    
    SymVec inv;
    inv.push_back(Sym(prod_token, args2));
    
    args.push_back( Sym(inv_token, inv) );

    return Sym(prod_token, args);
}

Sym operator-(Sym a, Sym b) {
    
    SymVec args;

    collect(sum_token, a, args);
    
    SymVec args2;
    collect(sum_token, b, args2);
    
    SymVec min;
    min.push_back(Sym(sum_token, args2));
    
    args.push_back( Sym(min_token, min) );

    return Sym(sum_token, args);
}

Sym operator-(Sym a) {
    return Sym(min_token, a);
}

Sym pow(Sym a, Sym b) {
    SymVec args;
    args.push_back(a);
    args.push_back(b);
    return Sym(pow_token, args);
}

Sym ifltz(Sym a, Sym b, Sym c) {
    SymVec args;
    args.push_back(a);
    args.push_back(b);
    args.push_back(c);
    return Sym(ifltz_token, args);
}

