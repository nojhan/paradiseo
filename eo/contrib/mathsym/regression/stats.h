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

class Mean {

    double n;
    double mean;

    public:
    Mean() : n(0), mean(0) {}
    
    void update(double v) {
	n++;
	double d = v - mean;
	mean += 1/n * d;
    }

    double get_mean() const { return mean; }
};

class Var {
    double n;
    double mean;
    double sumvar;

    public:
    Var() : n(0), mean(0), sumvar(0) {}

    void update(double v) {
	n++;
	double d = v - mean;
	mean += 1/n * d;
	sumvar += (n-1)/n * d * d;
    }

    double get_mean() const { return mean;           }
    double get_var()  const { return sumvar / (n-1); }
    double get_std()  const { return sqrt(get_var());  }
};

/** Single covariance between two variates */
class Cov {
    double n;
    double meana;
    double meanb;
    double sumcov;
    
    public:
    Cov() : n(0), meana(0), meanb(0), sumcov(0) {}

    void update(double a, double b) {
	++n;
	double da = a - meana;
	double db = b - meanb;

	meana += 1/n * da;
	meanb += 1/n * db;

	sumcov += (n-1)/n * da * db;
    }
    
    double get_meana() const { return meana; }
    double get_meanb() const { return meanb; }
    double get_cov()   const { return sumcov / (n-1); }
};

class CovMatrix {
    double n;
    std::vector<double> mean;
    std::vector< std::vector<double> > sumcov;

    public:
    CovMatrix(unsigned dim) : n(0), mean(dim), sumcov(dim , std::vector<double>(dim)) {}

    void update(const std::vector<double>& v) {
	n++;
	
	for (unsigned i = 0; i < v.size(); ++i) {
	    double d = v[i] - mean[i];
	    mean[i] += 1/n * d;

	    sumcov[i][i] += (n-1)/n * d * d;

	    for (unsigned j = i; j < v.size(); ++j) {
		double e = v[j] - mean[j]; // mean[j] is not updated yet
		
		double upd = (n-1)/n * d * e;

		sumcov[i][j] += upd;
		sumcov[j][i] += upd;
		
	    }
	}
	
    }
    
    double get_mean(int i) const { return mean[i]; }
    double get_var(int i ) const { return sumcov[i][i] / (n-1); }
    double get_std(int i) const  { return sqrt(get_var(i)); }
    double get_cov(int i, int j) const { return sumcov[i][j] / (n-1); }
    
};

