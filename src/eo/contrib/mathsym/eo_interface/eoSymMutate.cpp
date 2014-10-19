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

#include <eoSymMutate.h>
#include <FunDef.h>
#include <utils/eoRNG.h>

using namespace std;

std::pair<Sym, bool> do_mutate(Sym sym, double p, const LanguageTable& table) {
    
    bool changed = false;
    SymVec args = sym.args();
    if (rng.flip(p)) {
	token_t new_token = table.get_random_function(sym.token(), args.size());
	if (new_token != sym.token()) {
	    changed = true;
	    sym = Sym(new_token, args);
	}
    }

    for (unsigned i = 0; i < args.size(); ++i) {
	std::pair<Sym,bool> r = do_mutate(args[i], p, table);	
	changed |= r.second;
	if (r.second) 
	    args[i] = r.first;
    }

    if (changed)
	return std::make_pair(Sym(sym.token(), args), true);
    // else
    return std::make_pair(sym, false);
}
	
	
// these two can (should?) move to an impl file
bool mutate(Sym& sym, double p, const LanguageTable& table) {
    std::pair<Sym, bool> r = do_mutate(sym, p, table);
    sym = r.first;
    return r.second;
}


bool mutate_constants(Sym& sym, double stdev) {
    vector<double> values = get_constants(sym);

    if (values.empty()) {
	return false;
    }
    
    for (unsigned i = 0; i < values.size(); ++i) {
	values[i] += rng.normal() * stdev / values.size();
    }
    
    sym = set_constants(sym, values);
    return true;
}

