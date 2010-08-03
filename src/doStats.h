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

#ifndef _doStats_h
#define _doStats_h

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <eoPrintable.h>
#include <eoPop.h>

namespace ublas = boost::numeric::ublas;

class doStats : public eoPrintable
{
public:
    doStats();

    virtual void printOn(std::ostream&) const;

protected:
    double _n;
};

class doMean : public doStats
{
public:
    doMean();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_mean() const;

protected:
    double _mean;
};

class doVar : public doMean
{
public:
    doVar();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_var() const;
    double get_std() const;

protected:
    double _sumvar;
};

/** Single covariance between two variates */
class doCov : public doStats
{
public:
    doCov();

    virtual void update(double, double);
    virtual void printOn(std::ostream&) const;

    double get_meana() const;
    double get_meanb() const;
    double get_cov()   const;

protected:
    double _meana;
    double _meanb;
    double _sumcov;
};

class doCovMatrix : public doStats
{
public:
    doCovMatrix(unsigned dim);

    virtual void update(const std::vector<double>&);

    double get_mean(int) const;
    double get_var(int) const;
    double get_std(int) const;
    double get_cov(int, int) const;

protected:
    std::vector< double > _mean;
    std::vector< std::vector< double > > _sumcov;
};

class doHyperVolume : public doStats
{
public:
    doHyperVolume();

    virtual void update(double);
    virtual void printOn(std::ostream&) const;

    double get_hypervolume() const;

protected:
    double _hv;
};

template < typename EOT >
class doUblasCovMatrix : public doStats
{
public:
    typedef typename EOT::AtomType AtomType;

    virtual void update( const eoPop< EOT >& pop )
    {
	unsigned int p_size = pop.size(); // population size

	assert(p_size > 0);

	unsigned int s_size = pop[0].size(); // solution size

	assert(s_size > 0);

	ublas::matrix< AtomType > sample( p_size, s_size );

	for (unsigned int i = 0; i < p_size; ++i)
	    {
		for (unsigned int j = 0; j < s_size; ++j)
		    {
			sample(i, j) = pop[i][j];
		    }
	    }

	_varcovar.resize(s_size, s_size);

	// variance-covariance matrix are symmetric (and semi-definite positive),
	// thus a triangular storage is sufficient
	ublas::symmetric_matrix< AtomType, ublas::lower > var(s_size, s_size);

	// variance-covariance matrix computation : A * transpose(A)
	var = ublas::prod( sample, ublas::trans( sample ) );

	for (unsigned int i = 0; i < s_size; ++i)
	    {
		// triangular LOWER matrix, thus j is not going further than i
		for (unsigned int j = 0; j <= i; ++j)
		    {
			// we want a reducted covariance matrix
			_varcovar(i, j) = var(i, j) / p_size;
		    }
	    }

	//_varcovar = varcovar;

	_mean.resize(s_size);

        // unit vector
	ublas::scalar_vector< AtomType > u( p_size, 1 );

	// sum over columns
	ublas::vector< AtomType > mean = ublas::prod( ublas::trans( sample ), u );

	// division by n
	mean /= p_size;

	// copy results in the params std::vector
	std::copy(mean.begin(), mean.end(), _mean.begin());
    }

    const ublas::symmetric_matrix< AtomType, ublas::lower >& get_varcovar() const {return _varcovar;}

    const EOT& get_mean() const {return _mean;}

private:
    ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
    EOT _mean;
};

template < typename EOT >
class Cholesky : public doStats
{
public:
    typedef typename EOT::AtomType AtomType;

    virtual void update( const ublas::symmetric_matrix< AtomType, ublas::lower >& V)
    {
	unsigned int Vl = V.size1();

	assert(Vl > 0);

	unsigned int Vc = V.size2();

	assert(Vc > 0);

	_L.resize(Vl, Vc);

	unsigned int i,j,k;

	// first column
	i=0;

	// diagonal
	j=0;
	_L(0, 0) = sqrt( V(0, 0) );

	// end of the column
	for ( j = 1; j < Vc; ++j )
	    {
		_L(j, 0) = V(0, j) / _L(0, 0);
	    }

	// end of the matrix
	for ( i = 1; i < Vl; ++i )
	    { // each column

		// diagonal
		double sum = 0.0;

		for ( k = 0; k < i; ++k)
		    {
			sum += _L(i, k) * _L(i, k);
		    }

		assert( ( V(i, i) - sum ) > 0 );

		_L(i, i) = sqrt( V(i, i) - sum );

		for ( j = i + 1; j < Vl; ++j )
		    { // rows

			// one element
			sum = 0.0;

			for ( k = 0; k < i; ++k )
			    {
				sum += _L(j, k) * _L(i, k);
			    }

			_L(j, i) = (V(j, i) - sum) / _L(i, i);
		    }
	    }
    }

    const ublas::symmetric_matrix< AtomType, ublas::lower >& get_L() const {return _L;}

private:
    ublas::symmetric_matrix< AtomType, ublas::lower > _L;
};

#endif // !_doStats_h
