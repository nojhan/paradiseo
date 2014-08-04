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
//#include <eoSymLambdaMutate.h>
#include <eoSymCrossover.h>
#include <eoSymEval.h>
#include <eoOpContainer.h>
#include <eoDetTournamentSelect.h>
#include <eoMergeReduce.h>
#include <eoGenContinue.h>
#include <eoEasyEA.h>
#include <eoGeneralBreeder.h>

#include <utils/eoParser.h>
#include <utils/eoCheckPoint.h>
#include <utils/eoStat.h>
#include <utils/eoStdoutMonitor.h>
#include <utils/eoRNG.h>

using namespace std;

typedef EoSym<eoMinimizingFitness> EoType;

static int functions_added = 0;

void add_function(LanguageTable& table, eoParser& parser, string name, unsigned arity, token_t token, const FunDef& fun); 
void setup_language(LanguageTable& table, eoParser& parser);

template <class T>
T& select(bool check, T& a, T& b) { if (check) return a; return b; }

class eoBestIndividualStat : public eoSortedStat<EoType, string> {
    public: 
    eoBestIndividualStat() : eoSortedStat<EoType, string>("", "best individual") {}
    
    void operator()(const vector<const EoType*>& _pop)  {
	ostringstream os;
	os << (Sym) *_pop[0];
	value() = os.str();
    }
    
};

class AverageSizeStat : public eoStat<EoType, double> {
    public:
	AverageSizeStat() : eoStat<EoType, double>(0.0, "Average size population") {}
    
	void operator()(const eoPop<EoType>& _pop) {
	    double total = 0.0;
	    for (unsigned i = 0; i < _pop.size(); ++i) {
		total += _pop[i].size();
	    }
	    value() = total/_pop.size();
	}
};

class SumSizeStat : public eoStat<EoType, unsigned> {
    public:
	SumSizeStat() : eoStat<EoType, unsigned>(0u, "Number of subtrees") {}
    
	void operator()(const eoPop<EoType>& _pop) {
	    unsigned total = 0;
	    for (unsigned i = 0; i < _pop.size(); ++i) {
		total += _pop[i].size();
	    }
	    value() = total;
	}
};

class DagSizeStat : public eoStat<EoType, unsigned> {
    public:
	DagSizeStat() : eoStat<EoType, unsigned>(0u, "Number of distinct subtrees") {}

	void operator()(const eoPop<EoType>& _pop) {
	    value() = Sym::get_dag().size();
	}
};

int main(int argc, char* argv[]) {
   
    eoParser parser(argc, argv);
  
    /* Language */
    LanguageTable table;
    setup_language(table, parser);
   
    /* Data */
    
    eoValueParam<string> datafile = parser.createParam(string(""), "datafile", "Training data", 'd', string("Regression"), true); // mandatory 
    double train_percentage = parser.createParam(1.0, "trainperc", "Percentage of data used for training", 0, string("Regression")).value();
    
    /* Population */

    unsigned pop_size = parser.createParam(500u, "population-size", "Population Size", 'p', string("Population")).value();
  
    uint32_t seed = parser.createParam( uint32_t(time(0)), "random-seed", "Seed for rng", 'D').value();

    cout << "Seed " << seed << endl;
    rng.reseed(seed);
    
    double var_prob = parser.createParam(
	    0.9, 
	    "var-prob", 
	    "Probability of selecting a var vs. const when creating a terminal",
	    0,
	    "Population").value();

    
    double grow_prob = parser.createParam(
	    0.5,
	    "grow-prob",
	    "Probability of selecting 'grow' method instead of 'full' in initialization and mutation",
	    0,
	    "Population").value();
    
    unsigned max_depth = parser.createParam(
	    8u,
	    "max-depth",
	    "Maximum depth used in initialization and mutation",
	    0,
	    "Population").value();
	    
   
    bool use_uniform = parser.createParam(
	    false,
	    "use-uniform",
	    "Use uniform node selection instead of bias towards internal nodes (functions)",
	    0,
	    "Population").value();
	    
    double constant_mut_prob = parser.createParam(
	    0.1,
	    "constant-mut-rate",
	    "Probability of performing constant mutation",
	    0,
	    "Population").value();
    
    
    double subtree_mut_prob = parser.createParam(
	    0.2,
	    "subtree-mut-rate",
	    "Probability of performing subtree mutation",
	    0,
	    "Population").value();
    
    double node_mut_prob = parser.createParam(
	    0.2,
	    "node-mut-rate",
	    "Probability of performing node mutation",
	    0,
	    "Population").value();
    
/*    double lambda_mut_prob = parser.createParam(
	    1.0,
	    "lambda-mut-rate",
	    "Probability of performing (neutral) lambda extraction/expansion",
	    0,
	    "Population").value();
*/
    double subtree_xover_prob = parser.createParam(
	    0.4,
	    "xover-rate",
	    "Probability of performing subtree crossover",
	    0,
	    "Population").value();

    double homologous_prob = parser.createParam(
	    0.4,
	    "homologous-rate",
	    "Probability of performing homologous crossover",
	    0,
	    "Population").value();

    unsigned max_gens = parser.createParam(
	    50,
	    "max-gens",
	    "Maximum number of generations to run",
	    'g',
	    "Population").value();
    
    unsigned tournamentsize = parser.createParam(
	    5,
	    "tournament-size",
	    "Tournament size used for selection",
	    't',
	    "Population").value();

    unsigned maximumSize = parser.createParam(
	    -1u,
	    "maximum-size",
	    "Maximum size after crossover",
	    's',
	    "Population").value();
    
    unsigned meas_param = parser.createParam(
	    2u,
	    "measure",
	    "Error measure:\n\
		0 -> absolute error\n\
		1 -> mean squared error\n\
		2 -> mean squared error scaled (equivalent with correlation)\n\
		",
		'm',
		"Regression").value();
  
    
    ErrorMeasure::measure meas = ErrorMeasure::mean_squared_scaled;
    if (meas_param == 0) meas = ErrorMeasure::absolute;
    if (meas_param == 1) meas = ErrorMeasure::mean_squared;

    
    /* End parsing */
    if (parser.userNeedsHelp())
    {
	parser.printHelp(std::cout);
	return 1;
    }
    
    if (functions_added == 0) {
	cout << "ERROR: no functions defined" << endl;
	exit(1);
    }
    
    
    Dataset dataset;
    dataset.load_data(datafile.value());
    
    cout << "Data " << datafile.value() << " loaded " << endl;
   
    /* Add Variables */
    unsigned nvars = dataset.n_fields();
    for (unsigned i = 0; i < nvars; ++i) {
	table.add_function( SymVar(i).token(), 0);
    }
    
    TreeBuilder builder(table, var_prob);
    eoSymInit<EoType> init(builder, grow_prob, max_depth);
    
    eoPop<EoType> pop(pop_size, init);
    
    BiasedNodeSelector biased_sel;
    RandomNodeSelector random_sel;

    NodeSelector& node_selector = select<NodeSelector>(use_uniform, random_sel, biased_sel);
    
    //eoProportionalOp<EoType> genetic_operator;
    eoSequentialOp<EoType> genetic_operator;
    
    eoSymSubtreeMutate<EoType> submutate(builder, node_selector);
    genetic_operator.add( submutate, subtree_mut_prob);
   
    // todo, make this parameter, etc
    double std = 1.0;
    eoSymConstantMutate<EoType> constmutate(std);
    genetic_operator.add(constmutate, constant_mut_prob);
    
    eoSymNodeMutate<EoType>    nodemutate(table);
    genetic_operator.add(nodemutate, node_mut_prob);
   
//    eoSymLambdaMutate<EoType> lambda_mutate(node_selector);
//    genetic_operator.add(lambda_mutate, lambda_mut_prob); // TODO: prob should be settable
    
    //eoQuadSubtreeCrossover<EoType> quad(node_selector);
    eoSizeLevelCrossover<EoType> bin;//(node_selector);
    //eoBinSubtreeCrossover<EoType> bin(node_selector);
    genetic_operator.add(bin, subtree_xover_prob);
    
    eoBinHomologousCrossover<EoType> hom;
    genetic_operator.add(hom, homologous_prob);


    IntervalBoundsCheck check(dataset.input_minima(), dataset.input_maxima());
    ErrorMeasure measure(dataset, train_percentage, meas);

    eoSymPopEval<EoType> evaluator(check, measure, maximumSize);
    
    eoDetTournamentSelect<EoType> selectOne(tournamentsize);
    eoGeneralBreeder<EoType> breeder(selectOne, genetic_operator,1);
    eoPlusReplacement<EoType> replace;

    // Terminators
    eoGenContinue<EoType> term(max_gens);
    eoCheckPoint<EoType> checkpoint(term);
    
    eoBestFitnessStat<EoType> beststat;
    checkpoint.add(beststat);
   
    eoBestIndividualStat printer;
    AverageSizeStat avgSize;
    DagSizeStat dagSize;
    SumSizeStat sumSize;
    
    checkpoint.add(printer);
    checkpoint.add(avgSize);
    checkpoint.add(dagSize);
    checkpoint.add(sumSize);
    
    eoStdoutMonitor genmon;
    genmon.add(beststat);
    genmon.add(printer);
    genmon.add(avgSize);
    genmon.add(dagSize);
    genmon.add(sumSize);
    genmon.add(term); // add generation counter
    
    checkpoint.add(genmon);
    
    eoPop<EoType> dummy;
    evaluator(pop, dummy);
    
    eoEasyEA<EoType> ea(checkpoint, evaluator, breeder, replace);

    ea(pop); // run
    
}

void add_function(LanguageTable& table, eoParser& parser, string name, unsigned arity, token_t token, const FunDef& fun, bool all) {
    ostringstream desc;
    desc << "Enable function " << name << " arity = " << arity;
    bool enabled = parser.createParam(false, name, desc.str(), 0, "Language").value();
    
    if (enabled || all) {
	cout << "Func " << name << " enabled" << endl;
	table.add_function(token, arity);
	if (arity > 0) functions_added++;
    }
}

void setup_language(LanguageTable& table, eoParser& parser) {

    bool all = parser.createParam(false,"all", "Enable all functions").value();
    bool ratio = parser.createParam(false,"ratio","Enable rational functions (inv,min,sum,prod)").value();
    bool poly = parser.createParam(false,"poly","Enable polynomial functions (min,sum,prod)").value();
    
    // assumes that at this point all tokens are defined (none are zeroed out, which can happen with ERCs)
    vector<const FunDef*> lang = get_defined_functions(); 
    
    for (token_t i = 0; i < lang.size(); ++i) {
	
	if (lang[i] == 0) continue;
	
	bool is_poly = false;
	if (poly && (i == prod_token || i == sum_token || i == min_token) ) {
	    is_poly = true;
	}
	
	bool is_ratio = false;
	if (ratio && (is_poly || i == inv_token)) {
	    is_ratio = true;
	}
	
	const FunDef& fun = *lang[i]; 
	
	if (fun.has_varargs() ) {

	    for (unsigned j = fun.min_arity(); j < fun.min_arity() + 8; ++j) {
		if (j==1) continue; // prod 1 and sum 1 are useless
		ostringstream nm;
		nm << fun.name() << j;
		bool addanyway = (all || is_ratio || is_poly) && j == 2;
		add_function(table, parser, nm.str(), j, i, fun, addanyway);
	    }
	}
	else {
	    add_function(table, parser, fun.name(), fun.min_arity(), i, fun, all || is_ratio || is_poly);
	}
    }
}


