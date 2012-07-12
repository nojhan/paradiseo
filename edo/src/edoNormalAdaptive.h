
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
    Johann Dreo <johann.dreo@thalesgroup.com>
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#ifndef _edoNormalAdaptive_h
#define _edoNormalAdaptive_h

#include "edoDistrib.h"

#ifdef WITH_EIGEN

#include <Eigen/Dense>

template < typename EOT >
class edoNormalAdaptive : public edoDistrib< EOT >
{
public:
    //typedef EOT EOType;
    typedef typename EOT::AtomType AtomType;
    typedef Eigen::Matrix< AtomType, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix< AtomType, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    edoNormalAdaptive( unsigned int dim ) :
        _mean( Vector::Zero(dim) ),
        _C( Matrix::Identity(dim,dim) ),
        _B( Matrix::Identity(dim,dim) ),
        _D( Vector::Constant( dim, 1) ),
        _sigma(1.0),
        _p_c( Vector::Zero(dim) ),
        _p_s( Vector::Zero(dim) )
    {
        assert( dim > 0);
    }

    edoNormalAdaptive( unsigned int dim, 
            Vector mean,
            Matrix C,
            Matrix B,
            Vector D,
            double sigma,
            Vector p_c,
            Vector p_s
        ) :
        _mean( mean ),
        _C( C ),
        _B( B ),
        _D( D ),
        _sigma(sigma),
        _p_c( p_c ),
        _p_s( p_s )
    {
        assert( dim > 0);
        assert( _mean.innerSize() == dim );
        assert( _C.innerSize() == dim && _C.outerSize() == dim );
        assert( _B.innerSize() == dim && _B.outerSize() == dim );
        assert( _D.innerSize() == dim );
        assert( _sigma != 0.0 );
        assert( _p_c.innerSize() == dim );
        assert( _p_s.innerSize() == dim );
    }

    unsigned int size()
    {
        return _mean.innerSize();
    }

    Vector mean() const {return _mean;}
    Matrix covar() const {return _C;}
    Matrix coord_sys() const {return _B;}
    Vector scaling() const {return _D;}
    double sigma() const {return _sigma;}
    Vector path_covar() const {return _p_c;}
    Vector path_sigma() const {return _p_s;}

    void mean(       Vector m ) { _mean = m; }
    void covar(      Matrix c ) { _C = c; }
    void coord_sys(  Matrix b ) { _B = b; }
    void scaling(    Vector d ) { _D = d; }
    void sigma(      double s ) { _sigma = s; }
    void path_covar( Vector p ) { _p_c = p; }
    void path_sigma( Vector p ) { _p_s = p; }

private:
    Vector _mean; // 
    Matrix _C; // covariance matrix
    Matrix _B; // eigen vectors / coordinates system
    Vector _D; // eigen values / scaling
    double _sigma; // 
    Vector _p_c; // evolution path for C
    Vector _p_s; // evolution path for sigma
};

#endif // WITH_EIGEN

#endif // !_edoNormalAdaptive_h
