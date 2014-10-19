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


#include <LanguageTable.h>
#include <TreeBuilder.h>
#include <FunDef.h>
#include <Dataset.h>

#include <eoSymInit.h>
#include <eoSym.h>
#include <eoPop.h>
#include <eoSymMutate.h>
#include <eoSymCrossover.h>
#include <eoSymEval.h>

typedef EoSym<double> EoType;

int main() {
    
    LanguageTable table;
    table.add_function(sum_token, 2);
    table.add_function(prod_token, 2);
    table.add_function(inv_token, 1);
    table.add_function(min_token, 1);
    table.add_function( SymVar(0).token(), 0);
    
    table.add_function(tan_token, 1);
    
    table.add_function(sum_token, 0);
    table.add_function(prod_token, 0);
    
    TreeBuilder builder(table);

    eoSymInit<EoType> init(builder);
    
    eoPop<EoType> pop(10, init);
    
    for (unsigned i = 0; i < pop.size(); ++i) {
	// write out pretty printed
	cout << (Sym) pop[i] << endl;	
    }

    BiasedNodeSelector node_selector;
    eoSymSubtreeMutate<EoType> mutate1(builder, node_selector);
    eoSymNodeMutate<EoType> mutate2(table);
    
    cout << "****** MUTATION ************" << endl;

    for (unsigned i = 0; i < pop.size(); ++i) {
	
	cout << "Before  " << (Sym) pop[i] << endl;
	mutate1(pop[i]);
	cout << "After 1 " << (Sym) pop[i] << endl;
	mutate2(pop[i]);
	cout << "After 2 " << (Sym) pop[i] << endl;
    }
   
    cout << "****** CROSSOVER ***********" << endl;
    
    eoQuadSubtreeCrossover<EoType> quad(node_selector);
    eoBinSubtreeCrossover<EoType> bin(node_selector);
    eoBinHomologousCrossover<EoType> hom;

    for (unsigned i = 0; i < pop.size()-1; ++i) {
	cout << "Before    " << (Sym) pop[i] << endl;
	cout << "Before    " << (Sym) pop[i+1] << endl;
	
	hom(pop[i], pop[i+1]);
	
	cout << "After hom  " << (Sym) pop[i] << endl;
	cout << "After hom  " << (Sym) pop[i+1] << endl;
	
	
	quad(pop[i], pop[i+1]);

	cout << "After quad " << (Sym) pop[i] << endl;
	cout << "After quad " << (Sym) pop[i+1] << endl;
	
	bin(pop[i], pop[i+1]);
	
	cout << "After bin  " << (Sym) pop[i] << endl;
	cout << "After bin  " << (Sym) pop[i+1] << endl;
	
	cout << endl;
    }
  
    cout << "****** Evaluation **********" << endl;
    
    Dataset dataset;
    dataset.load_data("test_data.txt");
    IntervalBoundsCheck check(dataset.input_minima(), dataset.input_maxima());
    ErrorMeasure measure(dataset, 0.90, ErrorMeasure::mean_squared_scaled);

    eoSymPopEval<EoType> evaluator(check, measure, 20000);
    
    eoPop<EoType> dummy;
    evaluator(pop, dummy);

    for (unsigned i = 0; i < pop.size(); ++i) {
	cout << pop[i] << endl;
    }
    
}

