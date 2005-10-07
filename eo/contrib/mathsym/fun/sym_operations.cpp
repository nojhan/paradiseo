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

#include <FunDef.h>

using namespace std;

Sym simplify_constants(Sym sym) {

    SymVec args = sym.args();
    token_t token = sym.token();
    
    bool has_changed = false;
    bool all_constants = true;

    for (unsigned i = 0; i < args.size(); ++i) {
	
	Sym arg = simplify_constants(args[i]);
	
	if (arg != args[i]) {
	    has_changed = true;
	}
	args[i] = arg;

	
	all_constants &= is_constant(args[i].token());
    }
    
    if (args.size() == 0) return sym; // variable or constant
   
    
    if (all_constants) {
	// evaluate 
	
	vector<double> dummy;
	
	vector<double> vals(args.size());
	for (unsigned i = 0; i < vals.size(); ++i) {
	    vals[i] = eval(sym, dummy);
	}
	
	Sym result = SymConst( get_element(token).eval(vals, dummy) );
	
	
	return result;
    }

    if (has_changed) {
	return Sym(token, args);
    }

    return sym;
    
}

// currently only simplifies constants
Sym simplify(Sym sym) {
    
    return simplify_constants(sym);
    
}

