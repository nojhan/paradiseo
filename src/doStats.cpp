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

#include <cassert>
#include <cmath>

#include <vector>
#include <limits>

#include "doStats.h"

doStats::doStats()
    : _n(0)
{}

void doStats::printOn(std::ostream& _os) const
{
    _os << "Not implemented yet! ";
}

doMean::doMean()
    : _mean(0)
{}

void doMean::update(double v)
{
    _n++;

    double d = v - _mean;

    _mean += 1 / _n * d;
}

double doMean::get_mean() const
{
    return _mean;
}

void doMean::printOn(std::ostream& _os) const
{
    _os << get_mean();
}

doVar::doVar()
    : _sumvar(0)
{}

void doVar::update(double v)
{
    _n++;

    double d = v - _mean;

    _mean += 1 / _n * d;
    _sumvar += (_n - 1) / _n * d * d;
}

double doVar::get_var() const
{
    return _sumvar / (_n - 1);
}

double doVar::get_std() const
{
    return ::sqrt( get_var() );
}

void doVar::printOn(std::ostream& _os) const
{
    _os << get_var();
}

doCov::doCov()
    : _meana(0), _meanb(0), _sumcov(0)
{}

void doCov::update(double a, double b)
{
    ++_n;

    double da = a - _meana;
    double db = b - _meanb;

    _meana += 1 / _n * da;
    _meanb += 1 / _n * db;

    _sumcov += (_n - 1) / _n * da * db;
}

double doCov::get_meana() const
{
    return _meana;
}

double doCov::get_meanb() const
{
    return _meanb;
}

double doCov::get_cov() const
{
    return _sumcov / (_n - 1);
}

void doCov::printOn(std::ostream& _os) const
{
    _os << get_cov();
}

doCovMatrix::doCovMatrix(unsigned dim)
    : _mean(dim), _sumcov(dim, std::vector< double >( dim ))
{}

void doCovMatrix::update(const std::vector<double>& v)
{
    assert(v.size() == _mean.size());

    _n++;

    for (unsigned int i = 0; i < v.size(); ++i)
	{
	    double d = v[i] - _mean[i];

	    _mean[i] += 1 / _n * d;
	    _sumcov[i][i] += (_n - 1) / _n * d * d;

	    for (unsigned j = i; j < v.size(); ++j)
		{
		    double e = v[j] - _mean[j]; // _mean[j] is not updated yet

		    double upd = (_n - 1) / _n * d * e;

		    _sumcov[i][j] += upd;
		    _sumcov[j][i] += upd;
		}
	}
}

double doCovMatrix::get_mean(int i) const
{
    return _mean[i];
}

double doCovMatrix::get_var(int i) const
{
    return _sumcov[i][i] / (_n - 1);
}

double doCovMatrix::get_std(int i) const
{
    return ::sqrt( get_var(i) );
}

double doCovMatrix::get_cov(int i, int j) const
{
    return _sumcov[i][j] / (_n - 1);
}

doHyperVolume::doHyperVolume()
    : _hv(1)
{}

void doHyperVolume::update(double v)
{
    _hv *= ::sqrt(v);

    assert( _hv <= std::numeric_limits< double >::max() );
}

double doHyperVolume::get_hypervolume() const
{
    return _hv;
}

void doHyperVolume::printOn(std::ostream& _os) const
{
    _os << get_hypervolume();
}
