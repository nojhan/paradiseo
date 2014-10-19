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

#include <utils/eoRNG.h>

#include <FunDef.h>
#include <sym_compile.h>

#include <Dataset.h>
#include <ErrorMeasure.h>
#include <LanguageTable.h>
#include <BoundsCheck.h>
#include <TreeBuilder.h>

#include <iostream>

using namespace std;

void test_xover();

int main() {
    Dataset dataset;
    dataset.load_data("test_data.txt");
    
    cout << "Records/Fields " << dataset.n_records() << ' ' << dataset.n_fields() << endl;
   
    LanguageTable table;
    table.add_function(sum_token, 2);
    table.add_function(prod_token, 2);
    table.add_function(sum_token, 0);
    table.add_function(prod_token, 0);
    table.add_function(inv_token, 1);
    table.add_function(min_token, 1);
   
    for (unsigned i = 0; i < dataset.n_fields(); ++i) {
	table.add_function( SymVar(i).token(), 0);
    }
    
    TreeBuilder builder(table);
    
    IntervalBoundsCheck bounds(dataset.input_minima(), dataset.input_maxima() );
    ErrorMeasure measure(dataset, 1.0);
    
    
    unsigned n = 1000;
    unsigned k = 0;
    
    vector<Sym> pop;
    double sumsize = 0; 
    for (unsigned i = 0; i < n; ++i) {

	Sym sym = builder.build_tree(6, i%2);
	pop.push_back(sym);
	sumsize += sym.size();
    }
   
    cout << "Size " << sumsize/pop.size() << endl;
    
    // shuffle
    for (unsigned gen = 0; gen < 10; ++gen) {
	random_shuffle(pop.begin(), pop.end());
	for (unsigned i = 0; i < pop.size(); i+=2) {
	    
	    unsigned p1 = rng.random(pop[i].size());
	    unsigned p2 = rng.random(pop[i+1].size());

	    Sym a = insert_subtree(pop[i], p1, get_subtree(pop[i+1], p2));
	    Sym b = insert_subtree(pop[i+1], p2, get_subtree(pop[i], p1));

	    pop[i] = a;
	    pop[i+1] = b;
	    
	}
	cout << gen << ' ' << Sym::get_dag().size() << endl;
    }

    vector<Sym> oldpop;
    swap(pop,oldpop);
    for (unsigned i = 0; i < oldpop.size(); ++i) {
	Sym sym = oldpop[i];
	if (!bounds.in_bounds(sym)) {
	    k++;
	    continue;
	}
	pop.push_back(sym);
    }
    
    cout << "Done" << endl;
    
    // full compilation
    
    time_t start_time = time(0);
    time_t compile_time; 
    {
	multi_function f = compile(pop);
	compile_time = time(0);
	vector<double> out(pop.size());
        
        cout << "Compiled" << endl;
        	
	for (unsigned j = 0; j < dataset.n_records(); ++j) {
	    f(&dataset.get_inputs(j)[0], &out[0]);
	}
    }

    time_t end_time = time(0);

    cout << "Evaluated " << n-k << " syms in " << end_time - start_time << " seconds, compile took " << compile_time - start_time << " seconds" << endl;
    
    start_time = time(0);
    vector<single_function> funcs;
    compile(pop, funcs);
    compile_time = time(0);
    for (unsigned i = 0; i < pop.size(); ++i) {
	
	single_function f = funcs[i];
	for (unsigned j = 0; j < dataset.n_records(); ++j) {
	    f(&dataset.get_inputs(j)[0]);
	}
	
    }
     
    end_time = time(0);
    
    cout << "Evaluated " << n-k << " syms in " << end_time - start_time << " seconds, compile took " << compile_time - start_time << " seconds" << endl;
    return 0; // skip the 'slow' one-by-one method
    start_time = time(0);
    for (unsigned i = 0; i < pop.size(); ++i) {
	
	single_function f = compile(pop[i]);
	for (unsigned j = 0; j < dataset.n_records(); ++j) {
	    f(&dataset.get_inputs(j)[0]);
	}
	
    }
     
    end_time = time(0);
    
    cout << "Evaluated " << n-k << " syms in " << end_time - start_time << " seconds" << endl;
    
}

void test_xover() {
    Sym c = SymVar(0);
    Sym x = c + c * c + c;
    
    cout << c << endl;
    cout << x << endl;
    
    vector<Sym> pop;
    for (unsigned i = 0; i < x.size(); ++i) {
	for (unsigned j = 0; j < x.size(); ++j) {

	    Sym s = insert_subtree(x, i, get_subtree(x, j));
	    pop.push_back(s);
	    cout << i << ' ' << j << ' ' << s << endl;
	}
    }

    x = Sym();
    c = Sym();

    SymMap& dag = Sym::get_dag();

    for (SymMap::iterator it = dag.begin(); it != dag.end(); ++it) {
	Sym s(it);
	cout << s << ' ' << s.refcount() << endl;
    }
    
	    
    
}

