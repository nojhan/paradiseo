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

#ifndef SCALING_H_
#define SCALING_H_

#include "shared_ptr.h"

#include <valarray>
#include <iostream>
#include <string>

class TargetInfo;

class ScalingBase {
    public:
    
    virtual ~ScalingBase() {}
	
    std::valarray<double> apply(const std::valarray<double>& x) { 
	std::valarray<double> xtmp = x;
	transform(xtmp);
	return xtmp;
    }
	
    virtual double transform(double input) const = 0;
    virtual void transform(std::valarray<double>& inputs) const = 0;
    virtual std::ostream& print(std::ostream& os, std::string str) const = 0;
    virtual std::valarray<double> transform(const std::valarray<double>& inputs) const = 0;
};

typedef shared_ptr<ScalingBase> Scaling;

class LinearScaling : public ScalingBase {
    
    double a,b;
    
    public:
    LinearScaling() : a(0.0), b(1.0) {}
    LinearScaling(double _a, double _b) : a(_a), b(_b) {}

    double transform(double input) const { input *=b; input += a; return input; }
    void transform(std::valarray<double>& inputs) const { inputs *= b; inputs += a; }
    std::valarray<double> transform(const std::valarray<double>& inputs) const { 
	std::valarray<double> y = a + b * inputs;
	return y;
    }
    
    double intercept() const { return a; }
    double slope()     const { return b; }
    
    std::ostream& print(std::ostream& os, std::string str) const {
	os.precision(16);
	os << a << " + " << b << " * " << str;
	return os;
    }
};

class NoScaling : public ScalingBase{
    void transform(std::valarray<double>&) const {}
    double transform(double input) const { return input; }
    std::valarray<double> transform(const std::valarray<double>& inputs) const { return inputs; }
    std::ostream& print(std::ostream& os, std::string str) const { return os << str; }
};

extern Scaling slope(const std::valarray<double>& inputs, const TargetInfo& targets); // slope only
extern Scaling ols(const std::valarray<double>& inputs, const TargetInfo& targets);
extern Scaling wls(const std::valarray<double>& inputs, const TargetInfo& targets);
extern Scaling med(const std::valarray<double>& inputs, const TargetInfo& targets);

extern Scaling ols(const std::valarray<double>& inputs, const std::valarray<double>& outputs);

extern double mse(const std::valarray<double>& y, const TargetInfo& t);
extern double rms(const std::valarray<double>& y, const TargetInfo& t);
extern double mae(const std::valarray<double>& y, const TargetInfo& t);

// Todo Logistic Scaling

#endif


