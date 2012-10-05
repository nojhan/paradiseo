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

#include "Scaling.h"
#include "TargetInfo.h"

using namespace std;

Scaling slope(const std::valarray<double>& x, const TargetInfo& targets) {
    
    double xx = 0.0;
    double xy = 0.0;
    
    const valarray<double>& y = targets.targets();
    
    for (unsigned i = 0; i < x.size(); ++i) {
	xx += x[i] * x[i];
	xy += x[i] * y[i];
    }
    
    if (xx < 1e-7) return Scaling(new LinearScaling(0.0,0.0));
    
    double b = xy / xx;
   
    return Scaling(new LinearScaling(0.0, b));
    
}

// Still needs proper testing with non-trivial lambda
Scaling regularized_least_squares(const std::valarray<double>& inputs, const TargetInfo& targets, double lambda) {
    
    double n = inputs.size();
    
    valarray<double> x = inputs;

    double a,b,d;
    a=b=d=0;

    for (unsigned i = 0; i < n; ++i) {
	a += 1 + lambda;
	b += x[i];
	d += x[i] * x[i] + lambda;
    }
    
    //invert
    
    double ad_bc = a*d - b * b;
    // if ad_bc equals zero there's a problem
   
    if (ad_bc < 1e-17) return Scaling(new LinearScaling);
    
    double ai = d/ad_bc;
    double bi = -b/ad_bc;
    double di = a/ad_bc;
    double ci = bi;
    
    // Now multiply this inverted covariance matrix (C^-1) with x' * t
    
    std::valarray<double> ones = x;
    
    // calculate C^-1 * x' )
    for (unsigned i = 0; i < n; ++i) 
    {
	ones[i] = (ai + bi * x[i]);
	x[i]    = (ci + di * x[i]);
    }

    // results are in [ones, x], now multiply with y

    a = 0.0; // intercept
    b = 0.0; // slope
    
    const valarray<double>& t = targets.targets();
    
    for (unsigned i = 0; i < n; ++i)
    {
	a += ones[i] * t[i];
	b += x[i] * t[i];
    }
    
    return Scaling(new LinearScaling(a,b));
}

Scaling ols(const std::valarray<double>& y, const std::valarray<double>& t) {
    double n = y.size();
    
    double y_mean = y.sum() / n;
    double t_mean = t.sum() / n;
   
    std::valarray<double> y_var = (y - y_mean);
    std::valarray<double> t_var = (t - t_mean);
    std::valarray<double> cov = t_var * y_var;
    
    y_var *= y_var;
    t_var *= t_var;
    
    double sumvar = y_var.sum();
    
    if (sumvar == 0. || sumvar/n < 1e-7 || sumvar/n > 1e+7) // breakout when numerical problems are likely
	return Scaling(new LinearScaling(t_mean,0.));

    
    double b = cov.sum() / sumvar;
    double a = t_mean - b * y_mean;
    
    Scaling s = Scaling(new LinearScaling(a,b));

    return s;
}

Scaling ols(const std::valarray<double>& y, const TargetInfo& targets) {
    double n = y.size();
    
    double y_mean = y.sum() / n;
    
    std::valarray<double> y_var = (y - y_mean);
    std::valarray<double> cov = targets.tcov_part() * y_var;
    
    y_var *= y_var;

    double sumvar = y_var.sum();
    
    if (sumvar == 0. || sumvar/n < 1e-7 || sumvar/n > 1e+7) // breakout when numerical problems are likely
	return Scaling(new LinearScaling(targets.tmean(),0.));

    
    double b = cov.sum() / sumvar;
    double a = targets.tmean() - b * y_mean;
    
    if (!finite(b)) {
	
	cout << a << ' ' << b << endl;
	cout << sumvar << endl;
	cout << y_mean << endl;
	cout << cov.sum() << endl;
	exit(1);
    }
	
    Scaling s = Scaling(new LinearScaling(a,b));

    return s;
}


Scaling wls(const std::valarray<double>& inputs, const TargetInfo& targets) {
    
    std::valarray<double> x = inputs;
    const std::valarray<double>& w = targets.weights();
    
    unsigned n = x.size();
    // First calculate x'*W (as W is a diagonal matrix it's simply elementwise multiplication
    std::valarray<double> wx = targets.weights() * x;
    
    // Now x'*W is contained in [w,wx], calculate x' * W * x (the covariance)
    double a,b,d;
    a=b=d=0.0;
    
    for (unsigned i = 0; i < n; ++i)
    {
	a += w[i];
	b += wx[i];
	d += x[i] * wx[i];
    }
 
    //invert
    
    double ad_bc = a*d - b * b;
    // if ad_bc equals zero there's a problem
   
    if (ad_bc < 1e-17) return Scaling(new LinearScaling);
    
    double ai = d/ad_bc;
    double bi = -b/ad_bc;
    double di = a/ad_bc;
    double ci = bi;
    
    // Now multiply this inverted covariance matrix (C^-1) with x' * W * y
    
    // create alias to reuse the wx we do not need anymore
    std::valarray<double>& ones = wx;
    
    // calculate C^-1 * x' * W (using the fact that W is diagonal)
    for (unsigned i = 0; i < n; ++i) 
    {
	ones[i] = w[i]*(ai + bi * x[i]);
	x[i]    = w[i]*(ci + di * x[i]);
    }

    // results are in [ones, x], now multiply with y

    a = 0.0; // intercept
    b = 0.0; // slope
    
    const valarray<double>& t = targets.targets();
    
    for (unsigned i = 0; i < n; ++i)
    {
	a += ones[i] * t[i];
	b += x[i] * t[i];
    }
    
    return Scaling(new LinearScaling(a,b));
}


//Scaling med(const std::valarray<double>& inputs, const TargetInfo& targets);

double mse(const std::valarray<double>& y, const TargetInfo& t) {

    valarray<double> residuals = t.targets()-y;
    residuals *= residuals;
    double sz = residuals.size();
    if (t.has_weights()) {
	residuals *= t.weights();
	sz = 1.0;
    }
	
    return residuals.sum() / sz;
}

double rms(const std::valarray<double>& y, const TargetInfo& t) {
    return sqrt(mse(y,t));
}
    
double mae(const std::valarray<double>& y, const TargetInfo& t) {
    valarray<double> residuals = abs(t.targets()-y);
    if (t.has_weights()) residuals *= t.weights();
    return residuals.sum() / residuals.size();
}


/*
    double standard_error(const std::valarray<double>& y, const std::pair<double,double>& scaling) {
	double a = scaling.first;
	double b = scaling.second;
	double n = y.size();
	double se = sqrt( pow(a+b*y-current_set->targets,2.0).sum() / (n-2));
	
	double mean_y = y.sum() / n;
	double sxx = pow( y - mean_y, 2.0).sum();

	return se / sqrt(sxx);
    }
  
    double scaled_mse(const std::valarray<double>& y){
	std::pair<double,double> scaling;
	return scaled_mse(y,scaling);
    }
    
    double scaled_mse(const std::valarray<double>& y, std::pair<double, double>& scaling)
    {
	scaling = scale(y);
	
	double a = scaling.first;
	double b = scaling.second;
	
	std::valarray<double> tmp = current_set->targets - a - b * y;
	tmp *= tmp;
	
	if (weights.size())
	    return (weights * tmp).sum();

	return tmp.sum() / tmp.size();
    }
   
    double robust_mse(const std::valarray<double>& ny, std::pair<double, double>& scaling) {
	
	double smse = scaled_mse(ny,scaling);

	std::valarray<double> y = ny;
	// find maximum covariance case 
	double n = y.size();

	int largest = 0;
	
	{
	    double y_mean = y.sum() / n;
	
	    std::valarray<double> y_var = (y - y_mean);
	    std::valarray<double> cov = tcov * y_var;
	
	    std::valarray<bool> maxcov = cov == cov.max();
	
	    for (unsigned i = 0; i < maxcov.size(); ++i) {
		if (maxcov[i]) {
		    largest = i;
		    break;
		}
	    }
	}
	
	double y_mean = (y.sum() - y[largest]) / (n-1);
	y[largest] = y_mean; // dissappears from covariance calculation
	
	std::valarray<double> y_var = (y - y_mean);
	std::valarray<double> cov = tcov * y_var;
	y_var *= y_var;

	double sumvar = y_var.sum();
	
	if (sumvar == 0. || sumvar/n < 1e-7 || sumvar/n > 1e+7) // breakout when numerical problems are likely
	    return worst_performance();
	
	double b = cov.sum() / sumvar;
	double a = tmean - b * y_mean;
	
	std::valarray<double> tmp = current_set->targets - a - b * y;
	tmp[largest] = 0.0;
	tmp *= tmp;
	
	double smse2 = tmp.sum() / (tmp.size()-1);
	
	static std::ofstream os("smse.txt");
	os << smse << ' ' << smse2 << '\n';
	
	if (smse2 > smse) {
	    return worst_performance();
	    //std::cerr << "overfit? " << smse << ' ' << smse2 << '\n';
	}
	
	scaling.first = a;
	scaling.second = b;
	
	return smse2;
    }
    
    class Sorter {
	const std::valarray<double>& scores;
	public:
	    Sorter(const std::valarray<double>& _scores) : scores(_scores) {}
	    
	    bool operator()(unsigned i, unsigned j) const {
		return scores[i] < scores[j];
	    }
    };
    
    double coc(const std::valarray<double>& y) {
	std::vector<unsigned> indices(y.size());
	for (unsigned i = 0; i < y.size(); ++i) indices[i] = i;
	std::sort(indices.begin(), indices.end(), Sorter(y));
	
	const std::valarray<double>& targets = current_set->targets;
	
	double neg = 1.0 - targets[indices[0]];
	double pos = targets[indices[0]];
	
	double cumpos = 0;
	double cumneg = 0;
	double sum=0;
	
	double last_score = y[indices[0]];
	
	for(unsigned i = 1; i < targets.size(); ++i) {
	        
	    if (fabs(y[indices[i]] - last_score) < 1e-9) { // we call it tied
		pos += targets[indices[i]];
		neg += 1.0 - targets[indices[i]];
		
		if (i < targets.size()-1)
		    continue;
	    }
	    sum += pos * cumneg + (pos * neg) * 0.5;
	    cumneg += neg;
	    cumpos += pos;
	    pos = targets[indices[i]];
	    neg = 1.0 - targets[indices[i]];
	    last_score = y[indices[i]];
	}
	
	return sum / (cumneg * cumpos);
    }
   
    // iterative re-weighted least squares (for parameters.classification)
    double irls(const std::valarray<double>& scores, std::pair<double,double>& scaling) {
	const std::valarray<double>& t = current_set->targets;
	
	std::valarray<double> e(scores.size());
	std::valarray<double> u(scores.size()); 
	std::valarray<double> w(scores.size());
	std::valarray<double> z(scores.size());
	
	parameters.use_irls = false; parameters.classification=false;
	scaling = scale(scores);
	parameters.use_irls=true;parameters.classification=true;
	
	if (scaling.second == 0.0) return worst_performance(); 
	
	for (unsigned i = 0; i < 10; ++i) {
	    e = exp(scaling.first + scaling.second*scores);
	    u = e / (e + exp(-(scaling.first + scaling.second * scores)));
	    w = u*(1.-u);
	    z = (t-u)/w;
	    scaling = wls(scores, u, w);
	    //double ll = (log(u)*t + (1.-log(u))*(1.-t)).sum();
	    //std::cout << "Scale " << i << ' ' << scaling.first << " " << scaling.second << " LL " << 2*ll << std::endl;
	}

	// log-likelihood
	u = exp(scaling.first + scaling.second*scores) / (1 + exp(scaling.first + scaling.second*scores));
	double ll = (log(u)*t + (1.-log(u))*(1.-t)).sum();
	return 2*ll;
    }
*/
