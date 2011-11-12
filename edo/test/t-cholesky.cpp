
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
*/

#include <vector>
#include <cstdlib>
#include <iostream>

#include <eo>
#include <es.h>
#include <edo>

typedef eoReal< eoMinimizingFitness > EOT;
typedef edoNormalMulti<EOT> EOD;

std::ostream& operator<< (std::ostream& out, const ublas::symmetric_matrix< double, ublas::lower >& mat )
{
    for( unsigned int i=0; i<mat.size1(); ++i) {
        for( unsigned int j=0; j<=i; ++j) {
            out << mat(i,j) << "\t";
        } // columns
        out << std::endl;
    } // rows

    return out;
}

int main(int argc, char** argv)
{
    unsigned int N = 4;

    typedef edoSamplerNormalMulti<EOT,EOD>::Cholesky::MatrixType MatrixType;

    // a variance-covariance matrix of size N*N
    MatrixType V(N,N);

    // random covariance matrix
    for( unsigned int i=0; i<N; ++i) {
        V(i,i) = 1 + std::pow(rand(),2); // variance should be > 0
        for( unsigned int j=i+1; j<N; ++j) {
            V(i,j) = rand();
        }
    }

    std::cout << "Covariance matrix" << std::endl << V << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    edoSamplerNormalMulti<EOT,EOD>::Cholesky LLT( edoSamplerNormalMulti<EOT,EOD>::Cholesky::standard );
    edoSamplerNormalMulti<EOT,EOD>::Cholesky LLTa( edoSamplerNormalMulti<EOT,EOD>::Cholesky::absolute );
    edoSamplerNormalMulti<EOT,EOD>::Cholesky LDLT( edoSamplerNormalMulti<EOT,EOD>::Cholesky::robust );

    MatrixType L0 = LLT(V);
    std::cout << "LLT" << std::endl << L0 << std::endl;
    MatrixType V0 = ublas::prod( L0, ublas::trans(L0) );
    std::cout << "LLT covar" << std::endl << V0 << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    MatrixType L1 = LLTa(V);
    std::cout << "LLT abs" << std::endl << L1 << std::endl;
    MatrixType V1 = ublas::prod( L1, ublas::trans(L1) );
    std::cout << "LLT covar" << std::endl << V1 << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    
    MatrixType L2 = LDLT(V);
    MatrixType D2 = LDLT.diagonal();
    std::cout << "LDLT" << std::endl << L2 << std::endl;
    // ublas do not allow nested products, we should use a temporary matrix, 
    // thus the inline instanciation of a MatrixType
    // see: http://www.crystalclearsoftware.com/cgi-bin/boost_wiki/wiki.pl?Effective_UBLAS
    MatrixType V2 = ublas::prod( MatrixType(ublas::prod( L2, D2 )), ublas::trans(L2) );
    std::cout << "LDLT covar" << std::endl << V2 << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    
}
