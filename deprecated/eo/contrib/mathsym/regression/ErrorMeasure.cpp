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


#include <vector>
#include <valarray>

#include "MultiFunction.h"

#include "ErrorMeasure.h"
#include "Dataset.h"
#include "Sym.h"
#include "FunDef.h"
#include "sym_compile.h"
#include "TargetInfo.h"
#include "stats.h"

using namespace std;

#ifdef INTERVAL_DEBUG

#include <BoundsCheck.h>
#include <FunDef.h>

vector<double> none;
IntervalBoundsCheck bounds(none, none);

#endif



static double not_a_number = atof("nan");

class ErrorMeasureImpl {
    public:
	const Dataset& data;
	TargetInfo train_info;
	
	ErrorMeasure::measure measure;
	
	Scaling no_scaling;
	
	ErrorMeasureImpl(const Dataset& d, double t_p, ErrorMeasure::measure m) : data(d), measure(m) {
	    
#ifdef INTERVAL_DEBUG
	    bounds = IntervalBoundsCheck(d.input_minima(), d.input_maxima());
#endif

	    unsigned nrecords = d.n_records();
	    unsigned cases = unsigned(t_p * nrecords);
	    
	    valarray<double> t(cases);

	    for (unsigned i = 0; i < cases; ++i) {
		t[i] = data.get_target(i);
	    }

	    train_info = TargetInfo(t);
	    no_scaling = Scaling(new NoScaling);
	}
	
	ErrorMeasure::result eval(const valarray<double>& y) {
	    
	    ErrorMeasure::result result;
	    result.scaling = no_scaling;
	    
	    
	    switch(measure) {
		case ErrorMeasure::mean_squared:
		    result.error = pow(train_info.targets() - y, 2.0).sum() / y.size();
		    return result;
		case ErrorMeasure::absolute:
		    result.error = abs(train_info.targets() - y).sum() / y.size();
		    return result;
		case ErrorMeasure::mean_squared_scaled:
		    result.scaling = ols(y, train_info);
		    result.error  = pow(train_info.targets() - result.scaling->transform(y), 2.0).sum() / y.size();
		    return result;
		default: 
		    cerr << "Unknown measure encountered: " << measure << " " << __FILE__ << " " << __LINE__ << endl;
	    }
	    
	    return result;
	}   
	
	unsigned train_cases() const {
	    return train_info.targets().size();
	}

	vector<ErrorMeasure::result> multi_function_eval(const vector<Sym>& pop) {
	   
	    if (pop.size() == 0) return vector<ErrorMeasure::result>();
	    
	    multi_function all = compile(pop);
	    //MultiFunction all(pop);
	    std::vector<double> y(pop.size());
	    
	    Scaling noScaling = Scaling(new NoScaling);
	    
	    const std::valarray<double>& t = train_info.targets();
	    
	    cout << "Population size " << pop.size() << endl;
	    
	    if (measure == ErrorMeasure::mean_squared_scaled) {
		std::vector<Var> var(pop.size());
		std::vector<Cov> cov(pop.size());
	    
		Var vart;

		for (unsigned i = 0; i < t.size(); ++i) {
		    vart.update(t[i]);
		    
		    all(&data.get_inputs(i)[0], &y[0]); // evalutate
		    //all(data.get_inputs(i), y); // evalutate

		    for (unsigned j = 0; j < pop.size(); ++j) {
			var[j].update(y[j]);
			cov[j].update(y[j], t[i]);
		    }
		}
		
		std::vector<ErrorMeasure::result> result(pop.size());
		
		for (unsigned i = 0; i < pop.size(); ++i) {
		    
		    // calculate scaling
		    double b = cov[i].get_cov() / var[i].get_var();
		    
		    if (!finite(b)) {
			result[i].scaling = noScaling;
			result[i].error = vart.get_var(); // largest error
			continue;
		    }
		    
		    double a = vart.get_mean() - b * var[i].get_mean();
		    
		    result[i].scaling = Scaling( new LinearScaling(a,b));

		    // calculate error
		    double c = cov[i].get_cov();
		    c *= c;
		    
		    double err = vart.get_var() - c / var[i].get_var();
		    result[i].error = err; 
		    if (!finite(err)) {
			//cout << pop[i] << endl;
			cout << "b     " << b << endl;
			cout << "var t " << vart.get_var() << endl;
			cout << "var i " << var[i].get_var() << endl;
			cout << "cov   " << cov[i].get_cov() << endl;
			
			for (unsigned j = 0; j < t.size(); ++j) {
			    all(&data.get_inputs(i)[0], &y[0]); // evalutate
			    //all(data.get_inputs(j), y); // evalutate
			    
			    cout << y[i] << ' ' << ::eval(pop[i], data.get_inputs(j)) << endl;
			}
			
			exit(1);
		    }
		}
	
		return result;
	    }
	
	    
	    std::vector<double> err(pop.size()); 
	    
	    for (unsigned i = 0; i < train_cases(); ++i) {
		// evaluate
		all(&data.get_inputs(i)[0], &y[0]);
		//all(data.get_inputs(i), y);

		for (unsigned j = 0; j < pop.size(); ++j) {
		    double diff = y[j] - t[i];
		    if (measure == ErrorMeasure::mean_squared) { // branch prediction will probably solve this inefficiency
			err[j] += diff * diff;
		    } else {
			err[j] += fabs(diff);
		    }
		    
		}
		
	    }
	    
	    std::vector<ErrorMeasure::result> result(pop.size());

	    double n = train_cases();
	    for (unsigned i = 0; i < pop.size(); ++i) {
		result[i].error = err[i] / n;
		result[i].scaling = noScaling;
	    }

	    return result;
	    
	}

	vector<ErrorMeasure::result> single_function_eval(const vector<Sym> & pop) {
	    
	    vector<single_function> funcs(pop.size());
	    compile(pop, funcs); // get one function pointer for each individual

	    valarray<double> y(train_cases());
	    vector<ErrorMeasure::result> result(pop.size());
	    for (unsigned i = 0; i < funcs.size(); ++i) {
		for (unsigned j = 0; j < train_cases(); ++j) {
		    y[j] = funcs[i](&data.get_inputs(j)[0]);
		}
	
#ifdef INTERVAL_DEBUG
		//cout << "eval func " << i << " " << pop[i] << endl;
		pair<double, double> b = bounds.calc_bounds(pop[i]);
		
		// check if y is in bounds
		for (unsigned j = 0; j < y.size(); ++j) {
		    if (y[j] < b.first -1e-4 || y[j] > b.second + 1e-4 || !finite(y[j])) {
			cout << "Error " << y[j] << " not in " << b.first << ' ' << b.second << endl;
			cout << "Function " << pop[i] << endl;
			exit(1);
		    }
		}
#endif
		
		result[i] = eval(y);
	    }
	    
	    return result; 
	}
	
	vector<ErrorMeasure::result> calc_error(const vector<Sym>& pop) {

	    // first declone
#if USE_TR1
	    typedef std::tr1::unordered_map<Sym, unsigned, HashSym> HashMap;
#else
	    typedef hash_map<Sym, unsigned, HashSym> HashMap;
#endif	    
	    HashMap clone_map;
	    vector<Sym> decloned; 
	    decloned.reserve(pop.size());
	    
	    for (unsigned i = 0; i < pop.size(); ++i) {
		HashMap::iterator it = clone_map.find(pop[i]);

		if (it == clone_map.end()) { // new
		    clone_map[ pop[i] ] = decloned.size();
		    decloned.push_back(pop[i]);
		} 
		
	    }
	    
	    // evaluate 
	    vector<ErrorMeasure::result> dresult;
	    // currently we can only accumulate simple measures such as absolute and mean_squared
	    switch(measure) {
		case ErrorMeasure::mean_squared:
		case ErrorMeasure::absolute:
		    dresult = multi_function_eval(decloned);
		    break;
		case ErrorMeasure::mean_squared_scaled:
		    dresult = multi_function_eval(decloned);
		    break;
	    }
	    
	    vector<ErrorMeasure::result> result(pop.size());
	    for (unsigned i = 0; i < result.size(); ++i) {
		result[i] = dresult[ clone_map[pop[i]] ];
	    }
	
	    return result;
	}
	
};

ErrorMeasure::result::result() {
    error = 0.0;
    scaling = Scaling(0);
}

bool ErrorMeasure::result::valid() const {
    return isfinite(error);
}

ErrorMeasure::ErrorMeasure(const Dataset& data, double train_perc, measure meas) {
    pimpl = new ErrorMeasureImpl(data, train_perc, meas);
}

ErrorMeasure::~ErrorMeasure() { delete pimpl; }
ErrorMeasure::ErrorMeasure(const ErrorMeasure& that) { pimpl = new ErrorMeasureImpl(*that.pimpl); }


ErrorMeasure::result ErrorMeasure::calc_error(Sym sym) {
   
    single_function f = compile(sym);
    
    valarray<double> y(pimpl->train_cases());
     
    for (unsigned i = 0; i < y.size(); ++i) {

	y[i] = f(&pimpl->data.get_inputs(i)[0]);
	
	if (!finite(y[i])) {
	    result res;
	    res.scaling = Scaling(new NoScaling);
	    res.error = not_a_number;
	    return res;
	}
    }
   
    return pimpl->eval(y); 
}

vector<ErrorMeasure::result> ErrorMeasure::calc_error(const vector<Sym>& syms) {
    return pimpl->calc_error(syms);
    
}

double ErrorMeasure::worst_performance() const {
    
    if (pimpl->measure == mean_squared_scaled) {
	return pimpl->train_info.tvar();
    }
    
    return 1e+20; 
}

