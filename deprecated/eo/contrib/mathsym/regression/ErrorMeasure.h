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

#ifndef ERROR_MEASURE_H
#define ERROR_MEASURE_H

#include "Scaling.h"

class ErrorMeasureImpl;
class Sym;
class Dataset;

class ErrorMeasure {
    
    ErrorMeasureImpl* pimpl;
    
    public :

	enum measure {
	    absolute,
	    mean_squared,
	    mean_squared_scaled,
	};
	
	struct result {
	    double  error;
	    Scaling scaling;
	    
	    result();
	    bool valid() const;
	};
	
	ErrorMeasure(const Dataset& data, double train_perc, measure meas = mean_squared);
	
	~ErrorMeasure();
	ErrorMeasure(const ErrorMeasure& that);
	ErrorMeasure& operator=(const ErrorMeasure& that);
	
	result calc_error(Sym sym);

	std::vector<result> calc_error(const std::vector<Sym>& sym);
	
	double worst_performance() const;
};

#endif

