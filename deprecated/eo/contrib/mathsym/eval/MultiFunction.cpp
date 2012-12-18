#include <vector.h>


#include "MultiFunction.h"
#include "Sym.h"
#include "FunDef.h"

using namespace std;


typedef vector<double>::const_iterator data_ptr;
typedef vector<data_ptr> data_ptrs;
typedef data_ptrs::const_iterator arg_ptr;

#include "MultiFuncs.cpp"

typedef double (*fptr)( arg_ptr );

string print_function( fptr f) {
    if (f == multi_function::plus) return "+";
    if (f == multi_function::mult) return "*";
    if (f == multi_function::min) return "-";
    if (f == multi_function::inv) return "/";
    if (f == multi_function::exp) return "e";
    return "unknown";
}


struct Function {
    
    fptr function;
    arg_ptr args;

    double operator()() const { return function(args); }
};

static vector<Function> token_2_function;

Sym make_binary(Sym sym) {
    if (sym.args().size() == 2) return sym;
    SymVec args = sym.args();
    Sym an = args.back();
    args.pop_back();
    Sym nw = make_binary( Sym( sym.token(), args) );
    args.resize(2);
    args[0] = nw;
    args[1] = an;
    return Sym(sym.token(), args); 
}

class Compiler {
    
    public:

    enum func_type {constant, variable, function};
    
    typedef pair<func_type, unsigned> entry;

#if USE_TR1
	    typedef std::tr1::unordered_map<Sym, entry, HashSym> HashMap;
#else
	    typedef hash_map<Sym, entry, HashSym> HashMap;
#endif	    
   
    HashMap map;
   
    vector<double> constants;
    vector<unsigned> variables;
    vector< fptr > functions;
    vector< vector<entry> > function_args;
    
    unsigned total_args;
    
    vector<entry> outputs;
   
    Compiler() : total_args(0) {}
    
    entry do_add(Sym sym) {

	HashMap::iterator it = map.find(sym);

	if (it == map.end()) { // new entry
	    
	    token_t token = sym.token();

	    if (is_constant(token)) {
		constants.push_back( get_constant_value(token) ); // set value
		entry e = make_pair(constant, constants.size()-1);
		map.insert( make_pair(sym, e) );
		return e;
		
	    } else if (is_variable(token)) {
		unsigned idx = get_variable_index(token);
		variables.push_back(idx);
		entry e = make_pair(variable, variables.size()-1);
		map.insert( make_pair(sym, e) );
		return e;
	    } // else 
		
	    fptr f;
	    vector<entry> vec;
	    const SymVec& args = sym.args();
	    
	    switch (token) {
		case sum_token:
		    {
			if (args.size() == 0) {
			    return do_add( SymConst(0.0));
			}
			if (args.size() == 1) {
			    return do_add(args[0]);
			}
			if (args.size() == 2) {
			    vec.push_back(do_add(args[0]));
			    vec.push_back(do_add(args[1]));
			    f = multi_function::plus;
			    //cout << "Adding + " << vec[0].second << ' ' << vec[1].second << endl;
			    break;

			} else {
			    return do_add( make_binary(sym) );
			}
			
		    }
		case prod_token:
		    {
			if (args.size() == 0) {
			    return do_add( SymConst(1.0));
			}
			if (args.size() == 1) {
			    return do_add(args[0]);
			}
			if (args.size() == 2) {
			    vec.push_back(do_add(args[0]));
			    vec.push_back(do_add(args[1]));
			    f = multi_function::mult;
			    //cout << "Adding * " << vec[0].second << ' ' << vec[1].second << endl;
			    break;
			    

			} else {
			    return do_add( make_binary(sym) );
			}
		    }
		case sqr_token: 
		    {
			SymVec newargs(2);
			newargs[0] = args[0];
			newargs[1] = args[0];
		       return do_add( Sym(prod_token, newargs)); 
		    }
		default :
		    {
			if (args.size() != 1) {
			    cerr << "Unknown function " << sym << " encountered" << endl;
			    exit(1);
			}
			
			vec.push_back(do_add(args[0]));

			switch (token) {
			    case min_token: f = multi_function::min; break;
			    case inv_token: f = multi_function::inv; break;
			    case exp_token :f = multi_function::exp; break;
			    default :
				{
				    cerr << "Unimplemented token encountered " << sym << endl;
				    exit(1);
				}
			}
			
			//cout << "Adding " << print_function(f) << ' ' << vec[0].second << endl;
			
			
		    }

	    }
	    
	    total_args += vec.size();
	    function_args.push_back(vec);
	    functions.push_back(f);
	    
	    entry e = make_pair(function, functions.size()-1);
	    map.insert( make_pair(sym, e) );
	    return e;
	    
	}
	
	return it->second; // entry
    }
   
    void add(Sym sym) {
	entry e = do_add(sym);
	outputs.push_back(e);
    }
    
};

class MultiFunctionImpl {
    public:
	
    // input mapping
    vector<unsigned> input_idx;
    
    unsigned constant_offset;
    unsigned var_offset;

    // evaluation
    vector<double> data;
    vector<Function> funcs;
    data_ptrs args;	
    
    vector<unsigned> output_idx;
    
    MultiFunctionImpl() {}

    void clear() {
	input_idx.clear();
	data.clear();
	funcs.clear();
	args.clear();
	output_idx.clear();
	constant_offset = 0;
    }
    
    void eval(const double* x, double* y) {
	unsigned i;
	// evaluate variables
	for (i = constant_offset; i < constant_offset + input_idx.size(); ++i) {
	    data[i] = x[input_idx[i-constant_offset]];
	}

	for(; i < data.size(); ++i) {
	    data[i] = funcs[i-var_offset]();
	    //cout << i << " " << data[i] << endl;
	}

	for (i = 0; i < output_idx.size(); ++i) {
	    y[i] = data[output_idx[i]];
	}
    }

    void eval(const vector<double>& x, vector<double>& y) {
	eval(&x[0], &y[0]);
    }
    
    void setup(const vector<Sym>& pop) {
	
	clear(); 
	Compiler compiler;
	
	for (unsigned i = 0; i < pop.size(); ++i) {
	    Sym sym = (expand_all(pop[i]));
	    compiler.add(sym);
	}
	
	// compiler is setup so get the data
	constant_offset = compiler.constants.size();
	var_offset = constant_offset + compiler.variables.size();
	int n = var_offset + compiler.functions.size();

	data.resize(n);
	funcs.resize(compiler.functions.size());
	args.resize(compiler.total_args);
	
	// constants
	for (unsigned i = 0; i < constant_offset; ++i) {
	    data[i] = compiler.constants[i];
	    //cout << i << ' ' << data[i] << endl;
	}
	
	// variables
	input_idx = compiler.variables;

	//for (unsigned i = constant_offset; i < var_offset; ++i) {
	    //cout << i << " x" << input_idx[i-constant_offset] << endl;
	//}
	
	// functions
	unsigned which_arg = 0;
	for (unsigned i = 0; i < funcs.size(); ++i) {
	    
	    Function f;
	    f.function = compiler.functions[i];
	    
	    //cout << i+var_offset << ' ' << print_function(f.function);
	        
	    // interpret args
	    for (unsigned j = 0; j < compiler.function_args[i].size(); ++j) {
		
		Compiler::entry e = compiler.function_args[i][j];
		
		unsigned idx = e.second;
		
		switch (e.first) {
		    case Compiler::function: idx += compiler.variables.size();
		    case Compiler::variable: idx += compiler.constants.size();
		    case Compiler::constant: {}
		}

		args[which_arg + j] = data.begin() + idx;
		//cout << ' ' << idx << "(" << e.second << ")";
	    }
	    
	    //cout << endl;

	    f.args = args.begin() + which_arg;
	    which_arg += compiler.function_args[i].size();
	    funcs[i] = f;    
	}

	// output indices
	output_idx.resize(compiler.outputs.size());
	for (unsigned i = 0; i < output_idx.size(); ++i) {
	    output_idx[i] = compiler.outputs[i].second;
	    switch(compiler.outputs[i].first) {
		    case Compiler::function: output_idx[i] += compiler.variables.size();
		    case Compiler::variable: output_idx[i] += compiler.constants.size();
		    case Compiler::constant: {}
	    }
	    //cout << "out " << output_idx[i] << endl;
	}
    }
    
};  



MultiFunction::MultiFunction(const std::vector<Sym>& pop) : pimpl(new MultiFunctionImpl) {
    pimpl->setup(pop);
}

MultiFunction::~MultiFunction() { delete pimpl; }

void MultiFunction::operator()(const std::vector<double>& x, std::vector<double>& y) {
    pimpl->eval(x,y);
}

void MultiFunction::operator()(const double* x, double* y) {
    pimpl->eval(x,y);
}
