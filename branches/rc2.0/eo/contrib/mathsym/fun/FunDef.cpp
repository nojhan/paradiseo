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


#include <sstream>
#include "Sym.h"
#include "FunDef.h"
#include <LanguageTable.h>

using namespace std;
using namespace boost::numeric;

vector<const FunDef*> language; 

token_t add_function(FunDef* function) { 
    language.push_back(function); 
    return token_t(language.size()-1); 
}

const FunDef& get_element(token_t token) { return *language[token]; }

/* Printing */

string c_print(const Sym& sym) {
    return c_print(sym, vector<string>());
}

string c_print(const Sym& sym, const vector<string>& vars) {
    const SymVec& args = sym.args();
    vector<string> names(args.size());
    for (unsigned i = 0; i < args.size(); ++i) {
	names[i] = c_print(args[i], vars);
    }
    return language[sym.token()]->c_print(names, vars);
}

/* Evaluation */


double eval(const Sym& sym, const std::vector<double>& inputs) {
    return language[sym.token()]->eval(sym.args(), inputs);
}


/* Interval Logic */
Interval eval(const Sym& sym, const vector<Interval>& inputs) {
    const SymVec& args = sym.args();
    vector<Interval> interv(args.size());
    for (unsigned i = 0; i < args.size(); ++i) {
	interv[i] = eval(args[i], inputs);
    
	if (!valid(interv[i])) throw interval_error();
    }
    return language[sym.token()]->eval(interv, inputs);
}

/*  */
void add_function_to_table(LanguageTable& table, token_t token) {
    const FunDef& fundef = *language[token];
    
    if (fundef.has_varargs() == false) {
	table.add_function(token, fundef.min_arity());
    } else { // sum or prod (or min or max)
	table.add_function(token, 2);
    }
}


// by default it is eager
double FunDef::eval(const SymVec& args, const vector<double>& inputs) const {
    vector<double> values(args.size());
    for (unsigned i = 0; i < args.size(); ++i) {
	values[i] = ::eval(args[i], inputs);
    }

    return eval(values, inputs);
}

/* Variable Handling */
FunDef* make_var(int idx); // defined in FunDefs.h
static vector<token_t> var_token;

Sym SymVar(unsigned idx) {
    if (var_token.size() <= idx) {
	// it is new
	var_token.resize(idx+1, token_t(-1));
	var_token[idx] = add_function( make_var(idx) );
    } else if (var_token[idx] == token_t(-1)) {
	var_token[idx] = add_function( make_var(idx) );
    }
    return Sym(var_token[idx]);
}


/* Constant Handling */

struct HashDouble{
    size_t operator()(double val) const {
	unsigned long h = 0;
	char* s = (char*)&val;
	for (unsigned i=0 ; i<sizeof(double); ++i)
	    h = 5*h + s[i];
	return size_t(h);
    }
};

#if USE_TR1
typedef std::tr1::unordered_map<double, token_t> DoubleSet;
typedef std::tr1::unordered_map<Sym, token_t>    LambdaSet;
#else
typedef hash_map<double, token_t, HashDouble> DoubleSet;
typedef hash_map<Sym, token_t, HashSym> LambdaSet;
#endif

static DoubleSet doubleSet; // for quick checking if a constant already exists
static vector<double> token_value;

static LambdaSet lambdaSet;
static vector<Sym> token_lambda;

static std::vector<token_t> free_list;

void delete_val(token_t token) { // clean up the information about this value
     
    if (is_constant(token)) {
	//cout << "Deleting constant token " << token << endl;
	double value = token_value[token];
	doubleSet.erase(value);
	
	delete language[token];
	language[token] = 0;
	free_list.push_back(token);
    }
    else if (is_lambda(token)) {
	//cout << "Deleting lambda token " << token << endl;
	
	Sym expression = token_lambda[token];
	lambdaSet.erase(expression);
	
	delete language[token];
	language[token] = 0;
	free_list.push_back(token);
    }
}


FunDef* make_const(double value);

void extend_free_list();

Sym SymConst(double value) { 
   
    DoubleSet::iterator it = doubleSet.find(value);
    
    if (it != doubleSet.end()) {
	return Sym(it->second); // already exists
    }
       
    
    if (free_list.empty()) { // make space for tokens;
	extend_free_list();
    }

    token_t token = free_list.back();
    free_list.pop_back();
    //cout << "Creating constant with token " << token << endl; 
    assert(language[token] == 0);
    
    language[token] = make_const(value);
    
    doubleSet[value] = token;
    if (token_value.size() < token) token_value.resize(token+1);
    token_value[token] = value;
    
    return Sym(token);
}

/* LanguageTable depends on this one, XXX move somewhere safe.*/
#include <utils/eoRNG.h>
extern Sym default_const() { return SymConst(rng.normal()); }

/* The functions */
namespace {

class Var : public FunDef {
    public :
	unsigned idx;
	string default_str;

	Var(unsigned _idx) : idx(_idx) {
	    ostringstream os;
	    os << "x[" << idx << ']'; // CompiledCode expects this form
	    default_str = os.str();
	}
	
	double eval(const vector<double>& _, const vector<double>& inputs) const { return inputs[idx]; }
	double eval(const SymVec& _, const vector<double>& inputs) const { return inputs[idx]; }
	string c_print(const vector<string>& _, const vector<string>& names) const { 
	    if (names.empty()) {
		return default_str;
	    }
	    return names[idx]; 
	}

	Interval eval(const vector<Interval>& _, const vector<Interval>& inputs) const { 
	    return inputs[idx]; 
	}
	
	unsigned min_arity() const { return 0; }

	string name() const { return "var"; }
	
};

class Const : public FunDef {
    public:
	double value;
	string value_str;
	
	Const(double _value) : value(_value) {
	    ostringstream os;
	    os.precision(17);
	    os.setf(ios::showpoint);
	    os << '(' << value << ')';
	    value_str = os.str();	
	}


	double eval(const vector<double>& _, const vector<double>& inputs) const { return value; }
	double eval(const SymVec& _, const vector<double>& inputs) const { return value; }
	string c_print(const vector<string>& _, const vector<string>& names) const { 
	    return value_str; 
	}
	
	Interval eval(const vector<Interval>& _, const vector<Interval>& inputs) const { 
	    return Interval(value); 
	}

	unsigned min_arity() const { return 0; }

	string name() const { return "parameter"; }
};

} // namespace 
    
void get_constants(Sym sym, vector<double>& ret) {
    token_t token = sym.token();
    if (is_constant(token)) {
	double val = static_cast<const Const*>(language[token])->value;
	ret.push_back(val);
    }

    const SymVec& args = sym.args();
    for (unsigned i = 0; i < args.size(); ++i) {
	 get_constants(args[i], ret);
    }
    
}

double get_constant_value(token_t token) {
    return static_cast<const Const*>(language[token])->value;
}

/** Get out the values for all constants in the expression */
vector<double> get_constants(Sym sym) {
    vector<double> retval;
    get_constants(sym, retval);
    return retval;
}

/** Set the values for all constants in the expression. Vector needs to be the same size as the one get_constants returns 
 * The argument isn't touched, it will return a new sym with the constants set. */
Sym set_constants(Sym sym, vector<double>::const_iterator& it) {
    
    token_t token = sym.token();
    if (is_constant(token)) {
	return SymConst(*it++);
    }

    SymVec args = sym.args();
    for (unsigned i = 0; i < args.size(); ++i) {
	 args[i] = set_constants(args[i], it);
    }
    
    return Sym(token, args);
}

Sym set_constants(Sym sym, const vector<double>& constants) {
    vector<double>::const_iterator it = constants.begin();
    return set_constants(sym, it);
}

// Get functions out, excluding Const and Var
vector<const FunDef*> get_defined_functions() {
    vector<const FunDef*> res;
    for (unsigned i = 0; i < language.size(); ++i) {
	res.push_back(language[i]);

	if (is_constant(i) || is_variable(i)) {
	    res.back() = 0; // erase
	}
    }

    return res;
}

FunDef* make_var(int idx) { return new Var(idx); }
FunDef* make_const(double value) { return new Const(value); }

bool is_constant(token_t token) {
    const Const* cnst = dynamic_cast<const Const*>( language[token] );
    return cnst != 0;
}

bool is_variable(token_t token) {
    const Var* var = dynamic_cast<const Var*>( language[token] );
    return var != 0;
}

unsigned get_variable_index(token_t token) {
    const Var* var = static_cast<const Var*>( language[token] );
    return var->idx;
}

namespace {
class Lambda : public FunDef {
    public:
	Sym expression;
	int arity;
    
	Lambda(Sym expr, int arity_) : expression(expr), arity(arity_) {}
	
	double eval(const vector<double>& vals, const vector<double>& _) const {
	    return ::eval(expression, vals); 
	}
	
	string c_print(const vector<string>& args, const vector<string>& _) const {
	    return string("/*f*/") + ::c_print(expression, args) + string("/*eof*/");
	}
	
	Interval eval(const vector<Interval>& args, const vector<Interval>& _) const { 
	    return ::eval(expression, args);
	}

	unsigned min_arity() const { return arity; }
	string name() const { return "F"; }
	
};
    Sym normalize(Sym sym, SymVec& args) {
	// check if it's a variable
	token_t token = sym.token();
	const Var* var = dynamic_cast< const Var*>(language[token]);
    
	if (var != 0) {
	    for (unsigned i = 0; i < args.size(); ++i) {
		if (sym == args[i]) {
		    return SymVar(i); // replace with reference to arg
		}
	    }

	    // not replaced, add it
	    args.push_back(sym);
	    return SymVar(args.size()-1);
	    
	}
	
	SymVec a = sym.args();
	for (unsigned i = 0; i < a.size(); ++i) {
	    a[i] = normalize(a[i], args);
	}
	
	return Sym(token, a);
    }
}

bool is_lambda(token_t token) {
    const Lambda* lambda = dynamic_cast<const Lambda*>( language[token]); 
    return lambda != 0;
}

std::ostream& print_list(Sym sym, ostream& os) {
    os << sym.token() << ' ';
    
    const SymVec& args = sym.args();
    for (unsigned i = 0; i < args.size(); ++i) {
	print_list(args[i], os);
    }
    return os;
}

token_t new_lambda(Sym sym, int arity) {
    // check if already present

    LambdaSet::iterator it = lambdaSet.find(sym);
    if (it != lambdaSet.end()) {
	return it->second;
    }

    
    // new, insert
    Lambda* lambda = new Lambda(sym, arity);

    if (free_list.empty()) {
	extend_free_list();
    }
    
    token_t token = free_list.back();
    free_list.pop_back();
    language[token] = lambda;
    
    lambdaSet[sym] = token;
    if (token_lambda.size() <= token) token_lambda.resize(token+1);
    token_lambda[token] = sym;
    
    return token;
}

/* Compression */
typedef hash_map<Sym, unsigned, HashSym> OccMap;

void count_occurances(Sym sym, OccMap& occ) {
    occ[sym]++;
    const SymVec& args = sym.args();
    for (unsigned i = 0; i < args.size(); ++i) {
	count_occurances(args[i], occ);
    }
}

Sym create_lambda(Sym sym, OccMap& occ, unsigned nvars, vector<Sym>& args) {
    unsigned o = occ[sym];
    unsigned sz = sym.size();

    if (o * sz > o + sz + nvars || is_variable(sym.token()) ) {
	// check if it's already present
	for (unsigned i = 0; i < args.size(); ++i) {
	    if (args[i] == sym) {
		return SymVar(i);
	    }
	}
	// push_back
	args.push_back(sym);
	return SymVar(args.size()-1);
    }
    
    SymVec sym_args = sym.args();
    for (unsigned i = 0; i < sym_args.size(); ++i) {
	sym_args[i] = create_lambda(sym_args[i], occ, nvars, args);
    }

    return Sym(sym.token(), sym_args);
    
}

Sym compress(Sym sym) { 
    OccMap occ(sym.size());
    count_occurances(sym, occ);

    unsigned nvars = 0;
    for (OccMap::iterator it = occ.begin(); it != occ.end(); ++it) {
	if (is_variable(it->first.token())) nvars++;
    }
    
    SymVec args;
    Sym body = create_lambda(sym, occ, nvars, args);
    
    
    if (body.size() < sym.size()) {
	// see if the body can be compressed some more
	body = compress(body);
		
	token_t token = new_lambda(body, args.size());
	
	for (unsigned i = 0; i < args.size(); ++i) {
	    args[i] = compress(args[i]);
	}

	Sym result = Sym(token, args);
	return compress(result); // see if it can be compressed some more
    }
    
    return sym; 
}

Sym SymLambda(Sym expr) { return compress(expr); }

Sym expand(Sym expr, const SymVec& args) {
    
    const Var* var = dynamic_cast<const Var*>( language[expr.token()] );
    if (var != 0) {
	return args[var->idx];
    }

    SymVec expr_args = expr.args();
    for (unsigned i = 0; i < expr_args.size(); ++i) {
	expr_args[i] = expand(expr_args[i], args);
    }
    
    return Sym(expr.token(), expr_args);
}

Sym SymUnlambda(Sym sym) {
    Sym retval = sym;
    const Lambda* lambda = dynamic_cast<const Lambda*>( language[sym.token()] );
    if (lambda != 0) {
	retval = expand(lambda->expression, sym.args());
    }

    return retval;
}

Sym expand_all(Sym sym) {
    SymVec args = sym.args();
    for (unsigned i = 0; i < args.size(); ++i) {
	args[i] = expand_all(args[i]);
    }
    
    Sym nw = SymUnlambda( Sym(sym.token(), args) ); 
    
    if (nw != sym) {
	nw = expand_all(nw);
    }
    
    return nw;
}

namespace {

class Sum : public FunDef {

    public :
	
	double eval(const vector<double>& vals, const vector<double>& _) const { 
	    double res = 0;
	    for (unsigned i = 0; i < vals.size(); ++i) res += vals[i];
	    return res;
	}
	
	string c_print(const vector<string>& args, const vector<string>& _) const { 
	    if (args.empty()) { return "0.0"; }
	    
	    ostringstream os;
	    os << "(" << args[0];
	    for (unsigned i = 1; i < args.size(); ++i) {
		os << "+" << args[i];
	    }
	    os << ")";
	    return os.str();
	}
	
	Interval eval(const vector<Interval>& args, const vector<Interval>& inputs) const { 
	    Interval interv(0.0); 
	    for (unsigned i = 0; i < args.size(); ++i) {
		interv += args[i];
	    }
	    return interv;
	}

	unsigned min_arity() const { return 0; }
	bool has_varargs() const { return true; }

	string name() const { return "sum"; }
};


class Prod : public FunDef {

    public :
	
	double eval(const vector<double>& vals, const vector<double>& _) const { 
	    double res = 1;
	    for (unsigned i = 0; i < vals.size(); ++i) res *= vals[i];
	    return res;
	}
	
	string c_print(const vector<string>& args, const vector<string>& _) const { 
	    if (args.empty()) { return "1.0"; }
	    
	    ostringstream os;
	    os << "(" << args[0];
	    for (unsigned i = 1; i < args.size(); ++i) {
		os << "*" << args[i];
	    }
	    os << ")"; 

	    return os.str();
	}
	
	Interval eval(const vector<Interval>& args, const vector<Interval>& inputs) const { 
	    Interval interv(1.0);
	    for (unsigned i = 0; i < args.size(); ++i) {
		interv *= args[i];
	    }
	    return interv;
	}

	unsigned min_arity() const { return 0; }
	bool has_varargs() const { return true; }

	string name() const { return "prod"; }
};


class Power : public FunDef {
    public :
	double eval(const vector<double>& vals, const vector<double>& _) const {
	    return pow(vals[0], vals[1]);
	}

	string c_print(const vector<string>& args, const vector<string>& _) const {
	    return "pow(" + args[0] + ',' + args[1] + ')';
	}

	Interval eval(const vector<Interval>& args, const vector<Interval>& _) const {
	    Interval first = args[0];
	    Interval second = args[1];
	    Interval lg = log(first);
	    if (!valid(lg)) throw interval_error();
	    return exp(second * lg); 
	}

	unsigned min_arity() const { return 2; }

	string name() const { return "pow"; }
};

class IsNeg : public FunDef {

    public:
	double eval(const vector<double>& vals, const vector<double>& _) const {
	    if (vals[0] < 0.0) return vals[1];
	    return vals[2];
	}

	double eval(const Sym& sym, const vector<double>& inputs) const {
	    const SymVec& args = sym.args();
	    double arg0 = ::eval(args[0], inputs);
	    if (arg0 < 0.0) {
		return ::eval(args[1], inputs);
	    }
	    return ::eval(args[2], inputs);
	}

	string c_print(const vector<string>& args, const vector<string>& _) const {
	    return "((" + args[0] + "<0.0)?" + args[1] + ":" + args[2]+")";
	}
	
	Interval eval(const vector<Interval>& args, const vector<Interval>& _) const {
	    Interval a0 = args[0];
	    if (a0.upper() < 0.0) return args[1]; 
	    if (a0.lower() >= 0.0) return args[2];

	    return Interval( std::min(args[1].lower(), args[2].lower()), std::max(args[1].upper(), args[2].upper()));
	}

	unsigned min_arity() const { return 3; }

	string name() const { return "ifltz"; }
};

template <typename Func>
class Unary : public FunDef {
    
    Func un;
    
    double eval(const vector<double>& vals, const vector<double>& _) const { 
	return un(vals[0]);	
    }

    string c_print(const vector<string>& args, const vector<string>& _) const { 
	return un(args[0]);
    }

    Interval eval(const vector<Interval>& args, const vector<Interval>& _) const {
	return un(args[0]);
    }
    
    unsigned min_arity() const { return 1; }

    string name() const { return un.name(); }
    
};

struct Inv {
    double operator()(double val) const { return 1.0 / val; }
    string operator()(string v)   const { return "(1./" + v + ")"; }
    Interval operator()(Interval v) const { return 1.0 / v; }
    
    string name() const { return "inv"; }
};

struct Min {
    double operator()(double val) const { return -val; }
    string operator()(string v) const   { return "(-" + v + ")"; }
    Interval operator()(Interval v) const { return -v; }

    string name() const { return "min"; }
};

} // namespace

string prototypes = "double pow(double, double);";
string get_prototypes() { return prototypes; }
unsigned add_prototype(string str) { prototypes += string("double ") + str + "(double);"; return prototypes.size(); }

token_t add_function(FunDef* function, token_t where) {
    if (language.size() <= where) language.resize(where+1);
    language[where] = function;
    return 0;
}

namespace {

#define FUNCDEF(funcname) struct funcname##_struct { \
    double operator()(double val) const { return funcname(val); }\
    string operator()(string val) const  { return string(#funcname) + '(' + val + ')'; }\
    Interval operator()(Interval val) const { return funcname(val); }\
    string name() const { return string(#funcname); }\
};\
static const token_t funcname##_token_static = add_function( new Unary<funcname##_struct>, funcname##_token);\
unsigned funcname##_size = add_prototype(#funcname);

static token_t ssum_token  = add_function( new Sum , sum_token);
static token_t sprod_token = add_function( new Prod, prod_token);
static token_t sinv_token  = add_function( new Unary<Inv>, inv_token);
static token_t smin_token  = add_function( new Unary<Min>, min_token);
static token_t spow_token  = add_function( new Power, pow_token);
static token_t sifltz_token = add_function( new IsNeg, ifltz_token);

FUNCDEF(sin);
FUNCDEF(cos);
FUNCDEF(tan);
FUNCDEF(asin);
FUNCDEF(acos);
FUNCDEF(atan);

FUNCDEF(sinh);
FUNCDEF(cosh);
FUNCDEF(tanh);
FUNCDEF(asinh);
FUNCDEF(acosh);
FUNCDEF(atanh);

FUNCDEF(exp);
FUNCDEF(log);
} // namespace

double sqr(double x) { return x*x; }

namespace {
FUNCDEF(sqr);
FUNCDEF(sqrt);

const int buildInFunctionOffset = language.size();
} // namespace 

void add_tokens() {
    unsigned sz = language.size();
    language.resize(sz + sz+1); // double
    
    for (unsigned i = sz; i < language.size(); ++i) {
       free_list.push_back(i); 
    }
}

void extend_free_list() {
    // first check if we can clean up unused tokens;
    const vector<unsigned>& refcount = Sym::token_refcount();
    for (unsigned i = buildInFunctionOffset; i < refcount.size(); ++i) {
	if (language[i] == 0) continue;
	
	bool c = is_constant(i);
	bool l = is_lambda(i);
	
	if (refcount[i] == 0 && (c || l)) {
	    
	    if (c) {
		doubleSet.erase(token_value[i]);
	    }

	    if (l) {
		lambdaSet.erase(token_lambda[i]);	
	    }
	    
	    delete language[i];
	    language[i] = 0;
	    free_list.push_back(i);
	}
    }

    // if still empty, add new tokens
    if (free_list.empty()) {
	add_tokens();
    }
}


/* Serialization */
void write_raw(ostream& os, const Sym& sym) {
    token_t token = sym.token();
    const SymVec& args = sym.args();
    
    if (is_constant(token)) {
	os << "c" << language[token]->c_print(vector<string>(), vector<string>()); 
    } else {

	const Var* var = dynamic_cast<const Var*>( language[token] );
	
	if (var != 0) {
	    os << "v" << var->idx;
	} else {
	    os << "f" << token << ' ' << args.size();
	}
    }

    for (unsigned i = 0; i < args.size(); ++i) {
	write_raw(os, args[i]);	
    }
}

string write_raw(const Sym& sym) {
    
    ostringstream os;
    write_raw(os, sym);
    
    return os.str();
}

Sym read_raw(istream& is) {
    char id = is.get();

    switch (id) {
	case 'c' : 
	    {
		double val;
		is.get(); // skip '('
		is >> val;
		is.get(); // skip ')'
		return SymConst(val);
	    }
	case 'v' : 
	    {
		unsigned idx;
		is >> idx;
		return SymVar(idx);
	    }
	case 'f' :
	    {
		token_t token;
		unsigned arity;
		is >> token;
		is >> arity;
		SymVec args(arity);
		for (unsigned i = 0; i < arity; ++i) {
		    args[i] = read_raw(is);
		}

		return Sym(token, args);
	    }
	default : {
		      cerr << "Character = " << id << " Could not read formula from stream" << endl;
		      exit(1);
		  }
		    
    }

    return Sym();
}

Sym read_raw(string str) {
    istringstream is(str);
    return read_raw(is);
}

void read_raw(istream& is, Sym& sym) {
    sym = read_raw(is);
}

