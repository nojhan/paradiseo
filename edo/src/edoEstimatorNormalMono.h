/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoEstimatorNormalMono_h
#define _edoEstimatorNormalMono_h

#include "edoEstimator.h"
#include "edoNormalMono.h"

template < typename EOT >
class edoEstimatorNormalMono : public edoEstimator< edoNormalMono< EOT > >
{
public:
    typedef typename EOT::AtomType AtomType;

    class Variance
    {
    public:
	Variance() : _sumvar(0){}

	void update(AtomType v)
	{
	    _n++;

	    AtomType d = v - _mean;

	    _mean += 1 / _n * d;
	    _sumvar += (_n - 1) / _n * d * d;
	}

	AtomType get_mean() const {return _mean;}
	AtomType get_var() const {return _sumvar / (_n - 1);}
	AtomType get_std() const {return sqrt( get_var() );}

    private:
	AtomType _n;
	AtomType _mean;
	AtomType _sumvar;
    };

public:
    edoNormalMono< EOT > operator()(eoPop<EOT>& pop)
    {
	unsigned int popsize = pop.size();
	assert(popsize > 0);

	unsigned int dimsize = pop[0].size();
	assert(dimsize > 0);

	std::vector< Variance > var( dimsize );

	for (unsigned int i = 0; i < popsize; ++i)
	    {
		for (unsigned int d = 0; d < dimsize; ++d)
		    {
			var[d].update( pop[i][d] );
		    }
	    }

	EOT mean( dimsize );
	EOT variance( dimsize );

	for (unsigned int d = 0; d < dimsize; ++d)
	    {
		mean[d] = var[d].get_mean();
		variance[d] = var[d].get_var();
	    }

	return edoNormalMono< EOT >( mean, variance );
    }
};

#endif // !_edoEstimatorNormalMono_h
