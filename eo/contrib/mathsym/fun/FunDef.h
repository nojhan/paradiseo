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

#include "Sym.h"
#include "Interval.h"

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

extern std::vector<const FunDef*> get_defined_functions();
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
inline std::ostream& operator<<(std::ostream& os, Sym sym) { return os << c_print(sym); }

/** Create constant with this value, memory is managed. If reference count drops to zero value is deleted.  */
extern Sym SymConst(double value);

/** Get out the values for all constants in the expression */
std::vector<double> get_constants(Sym sym);

/** Set the values for all constants in the expression. Vector needs to be the same size as the one get_constants returns 
 * The argument isn't touched, it will return a new sym with the constants set. */
Sym set_constants(Sym sym, const std::vector<double>& constants);

/** check if a token is constant (for instance Sym sym; is_constant(sym.token()); ) */
extern bool is_constant(token_t token);

/** Create variable */
extern Sym SymVar(unsigned idx);

/* Add function to the language table (and take a guess at the arity) */
class LanguageTable;
extern void add_function_to_table(LanguageTable& table, token_t token);

// token names
extern const token_t sum_token;
extern const token_t prod_token;
extern const token_t inv_token;
extern const token_t min_token;
extern const token_t pow_token;
extern const token_t ifltz_token;

#define HEADERFUNC(name) extern const token_t name##_token;\
    inline Sym name(Sym arg) { return Sym(name##_token, arg); }

/* This defines the tokens: sin_token, cos_token, etc. */
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

extern std::string get_prototypes();

// reading and writing in internal format
extern std::string write_raw(const Sym& sym);
extern void write_raw(ostream& os, const Sym& sym);
extern Sym read_raw(std::string str);
extern Sym read_raw(std::istream& is); 
extern void read_raw(std::istream& is, Sym& sym); 

#include "SymOps.h"

#endif

