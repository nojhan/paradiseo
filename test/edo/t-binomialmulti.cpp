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

Copyright (C) 2013 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include <paradiseo/eo.h>
#include <paradiseo/edo.h>
#include <paradiseo/eo/ga.h> // for Bools


#ifdef WITH_EIGEN
#include <Eigen/Dense>
#endif // WITH_EIGEN

#ifdef WITH_EIGEN
// NOTE: a typedef on eoVector does not work, because of readFrom on a vector AtomType
// typedef eoVector<eoMinimizingFitness, std::vector<bool> > Bools;
class Bools : public std::vector<std::vector<bool> >, public EO<double>
{
    public:
        typedef std::vector<bool> AtomType;
};
#endif

int main(int ac, char** av)
{

    #ifdef WITH_EIGEN
        eoParser parser(ac, av);

        std::string        section("Algorithm parameters");
        unsigned int popsize = parser.createParam((unsigned int)100000, "popSize", "Population Size", 'P', section).value(); // P
        unsigned int rows = parser.createParam((unsigned int)2, "lines", "Lines number", 'l', section).value(); // l
        unsigned int cols = parser.createParam((unsigned int)3, "columns", "Columns number", 'c', section).value(); // c
        double proba = parser.createParam((double)0.5, "proba", "Probability to estimate", 'b', section).value(); // b

        if( parser.userNeedsHelp() ) {
                parser.printHelp(std::cout);
                exit(1);
        }

        make_help(parser);

        std::cout << "Init distrib" << std::endl;
        Eigen::MatrixXd initd = Eigen::MatrixXd::Constant(rows,cols,proba);
        edoBinomialMulti<Bools> distrib( initd );
        std::cout << distrib << std::endl;

        edoEstimatorBinomialMulti<Bools> estimate;
        edoSamplerBinomialMulti<Bools> sample;

        std::cout << "Sample a pop from the init distrib" << std::endl;
        eoPop<Bools> pop; pop.reserve(popsize);
        for( unsigned int i=0; i < popsize; ++i ) {
            pop.push_back( sample( distrib ) );
        }

        std::cout << "Estimate a distribution from the sampled pop" << std::endl;
        distrib = estimate( pop );
        std::cout << distrib << std::endl;

        std::cout << "Estimated initial proba = " << distrib.mean() << std::endl;
    #else
        #ifdef WITH_BOOST
        #pragma message "WARNING: there is no Boost::uBLAS implementation of t-binomialmulti, build WITH_EIGEN if you need it."
        #endif
    #endif // WITH_EIGEN
    return 0;
}
