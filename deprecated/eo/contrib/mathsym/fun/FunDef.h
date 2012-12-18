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

#ifndef FUNCTION_DEF_H_
#define FUNCTION_DEF_H_

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "sym/Sym.h"
#include "eval/Interval.h"

class FunDef {
    public:
    
    virtual ~FunDef() {}
    
    // (possibly) lazy evaluation function, default implementation calls 'eager' eval
    virtual double eval(const SymVec&   args, const std::vector<double>& inputs) const;

    // eager evaluation function 
    virtual double eval(const std::vector<double>& args, const std::vector<double>& inputs) const = 0;
   
    // interval evaluation
    virtual Interval eval(const std::vector<Interval>& args, const std::vector<Interval>& inputs) const = 0; 
    
    // prints 'c' like code
    virtual std::string c_print(const std::vector<std::string>&   names, const std::vector<std::string>& names) const = 0;

    virtual unsigned min_arity() const = 0;
    virtual bool has_varargs() const { return false; } // sum, prod, min, max are variable arity

    virtual std::string name() const = 0;
    
    protected:
    
};

/** Gets out all function that are defined (excluding constants and variables) */
extern std::vector<const FunDef*> get_defined_functions();

/** Gets a specific function (including vars and constants) out */
extern const FunDef& get_element(token_t token);

/** Single case evaluation */
extern double      eval(const Sym& sym, const std::vector<double>& inputs);

/** Static analysis through interval arithmetic */
extern Interval eval(const Sym& sym, const std::vector<Interval>& inputs);

/** Pretty printers, second version allows setting of variable names */
extern std::string c_print(const Sym& sym);

/** Pretty printers, allows setting of variable names */
extern std::string c_print(const Sym& sym, const std::vector<std::string>& var_names);

/** Pretty printer streamer */
inline std::ostream& operator<<(std::ostream& os, const Sym& sym) { return os << c_print(sym); }

/* Support for Ephemeral Random Constants (ERC) */

/** Create constant with this value, memory is managed. If reference count drops to zero value is deleted.  */
extern Sym SymConst(double value);
/** Create variable */
extern Sym SymVar(unsigned idx);

/** Create 'lambda expression; 
 * This is a neutral operation. It will replace
 * all variables in the expression by arguments,
 * wrap the expression in a Lambda function
 * and returns a tree applying the lambda function
 * to the original variable.
 *
 * A call like SymLambda( SymVar(1) + SymVar(1) * 3.1) will result in 
 * a Lambda function (a0 + a0 * 3.1) with one argument: SymVar(1)*/

extern Sym SymLambda(Sym expression);

extern Sym SymUnlambda(Sym sym);

/** Expands all lambda expressions inline */
extern Sym expand_all(Sym sym);
extern Sym compress(Sym sym);

/** Get out the values for all constants in the expression */
std::vector<double> get_constants(Sym sym);

/** Set the values for all constants in the expression. Vector needs to be the same size as the one get_constants returns 
 * The argument isn't touched, it will return a new sym with the constants set. */
Sym set_constants(Sym sym, const std::vector<double>& constants);

/** check if a token is a constant */
extern bool is_constant(token_t token);
extern double get_constant_value(token_t token);
/** check if a token is a variable */
extern bool is_variable(token_t token);
extern unsigned get_variable_index(token_t token);

/** check if a token is a user/automatically defined function */
extern bool is_lambda(token_t token);


/** simplifies a sym (sym_operations.cpp) Currently only simplifies constants */
extern Sym simplify(Sym sym);

/** differentiates a sym to a token (sym_operations.cpp)
 * The token can be a variable or a constant 
*/
extern Sym differentiate(Sym sym, token_t dx);
struct differentiation_error{}; // thrown in case of ifltz

/* Add function to the language table (and take a guess at the arity) */
class LanguageTable;
extern void add_function_to_table(LanguageTable& table, token_t token);

enum {
    sum_token,
    prod_token,
    inv_token,
    min_token,
    pow_token,
    ifltz_token,
    sin_token, cos_token, tan_token,
    asin_token, acos_token, atan_token,
    sinh_token, cosh_token, tanh_token,
    acosh_token, asinh_token, atanh_token,
    exp_token, log_token,
    sqr_token, sqrt_token
};

/* Defition of function overloads: for example, this define the function 'Sym sin(Sym)' */

#define HEADERFUNC(name) inline Sym name(Sym arg) { return Sym(name##_token, arg); }

/* This defines the tokens: sin_token, cos_token, etc. */
HEADERFUNC(inv);
HEADERFUNC(sin);
HEADERFUNC(cos);
HEADERFUNC(tan);
HEADERFUNC(asin);
HEADERFUNC(acos);
HEADERFUNC(atan);

HEADERFUNC(sinh);
HEADERFUNC(cosh);
HEADERFUNC(tanh);
HEADERFUNC(asinh);
HEADERFUNC(acosh);
HEADERFUNC(atanh);

HEADERFUNC(exp);
HEADERFUNC(log);

HEADERFUNC(sqr);
HEADERFUNC(sqrt);

/* Get the prototype functions out, this is needed for compilation */
extern std::string get_prototypes();

// reading and writing in internal format, no parser for symbolic functions implemented yet
extern std::string write_raw(const Sym& sym);
extern void write_raw(std::ostream& os, const Sym& sym);
extern Sym read_raw(std::string str);
extern Sym read_raw(std::istream& is); 
extern void read_raw(std::istream& is, Sym& sym); 

#include "SymOps.h"

#endif

