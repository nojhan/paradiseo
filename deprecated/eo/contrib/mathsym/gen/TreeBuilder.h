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

#ifndef TREEBUILDER_H_
#define TREEBUILDER_H_

#include "sym/Sym.h"
#include "LanguageTable.h"

class TreeBuilder {
    const LanguageTable& table;

    // probability of selecting a var versus a const when the choice boils down to selecting a terminal
    double vcprob;
    
    Sym make_terminal() const;
    public:

    TreeBuilder(const LanguageTable& t) : table(t), vcprob(0.9) {};
    TreeBuilder(const LanguageTable& t, double vc) : table(t), vcprob(vc) {};
    
    void set_var_vs_const_probability(double p) { vcprob = p; }
    
    Sym build_tree(unsigned max_depth, bool grow) const;

    void build_tree(Sym& tree, unsigned max_depth, bool grow) const { tree = build_tree(max_depth, grow); }
    
};

#endif

