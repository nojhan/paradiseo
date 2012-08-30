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
    
    if (args.size() == 0) {
	
	if (sym.token() == sum_token) return SymConst(0.0);
	if (sym.token() == prod_token) return SymConst(1.0);
	
	return sym; // variable or constant
    }
    
    if (all_constants) {
	// evaluate 
	
	vector<double> dummy;
	
	double v = ::eval(sym, dummy);
	
	Sym result = SymConst(v);
	
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

Sym derivative(token_t token, Sym x) {
    Sym one = Sym(prod_token);
    
    switch (token) {
	case inv_token : return Sym(inv_token, sqr(x));
	
	case sin_token : return -cos(x);
	case cos_token : return sin(x);
	case tan_token : return one + sqr(tan(x));
			 
	case asin_token : return inv( sqrt(one - sqr(x)));
	case acos_token:  return -inv( sqrt(one - sqr(x)));
	case atan_token : return inv( sqrt(one + sqr(x)));
	
	case cosh_token : return -sinh(x);
	case sinh_token : return cosh(x);
	case tanh_token : return one - sqr( tanh(x) );
	
	case asinh_token : return inv( sqrt( one + sqr(x) ));
	case acosh_token : return inv( sqrt(x-one) * sqrt(x + one)  );
	case atanh_token : return inv(one - sqr(x));
			 
	case exp_token : return exp(x);
	case log_token : return inv(x);

	case sqr_token : return SymConst(2.0) * x;
	case sqrt_token : return SymConst(0.5) * inv( sqrt(x));
	default :
	    throw differentiation_error();
    }
    
    return x;
}

extern Sym differentiate(Sym sym, token_t dx) {
    
    token_t token = sym.token();
    
    Sym zero = Sym(sum_token);
    Sym one  = Sym(prod_token);
    
    if (token == dx) {
	return one;
    }
    
    SymVec args = sym.args();

    if (args.size() == 0) { // df/dx with f != x
	return zero;
    }
    
    switch (token) {
	
	case sum_token: 
	    {
		for (unsigned i = 0; i < args.size(); ++i) {
		    args[i] = differentiate(args[i], dx);
		}

		if (args.size() == 1) return args[0];
		return Sym(sum_token, args);
	    }
	case min_token : 
	    {
		return -differentiate(args[0],dx);
	    }
	case prod_token: 
	    {
		if (args.size() == 1) return differentiate(args[0], dx);
		
		if (args.size() == 2) {
		    return args[0] * differentiate(args[1], dx) + args[1] * differentiate(args[0], dx);
		}
		// else 
		Sym c = args.back();
		args.pop_back();
		Sym f = Sym(prod_token, args);
		Sym df = differentiate( f, dx);

		return c * df + f * differentiate(c,dx);
	    }
	case pow_token : 
	    {
		return pow(args[0], args[1]) * args[1] * inv(args[0]);
	    }
	case ifltz_token : 
	    { // cannot be differentiated
		throw differentiation_error(); // TODO define proper exception
	    }
	    
	default: // unary function: apply chain rule
	    {
		Sym arg = args[0];
		return derivative(token, arg) * differentiate(arg, dx);
	    }
    }
    
}
