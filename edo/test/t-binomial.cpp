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

#include <eo>
#include <edo>
#include <ga.h> // for Bools

template<class T>
void print( T v, std::string sep=" " )
{
    for( typename T::iterator it=v.begin(); it!=v.end(); ++it ) {
        std::cout << *it << sep;
    }
    std::cout << std::endl;
}

int main(int ac, char** av)
{
    typedef eoBit<double> Bools;

    eoParser parser(ac, av);

    std::string        section("Algorithm parameters");
    unsigned int popsize = parser.createParam((unsigned int)100000, "popSize", "Population Size", 'P', section).value(); // P
    unsigned int dim = parser.createParam((unsigned int)20, "dim", "Dimension size", 'd', section).value(); // d
    double proba = parser.createParam((double)0.5, "proba", "Probability to estimate", 'b', section).value(); // b

    if( parser.userNeedsHelp() ) {
            parser.printHelp(std::cout);
            exit(1);
    }

    make_help(parser);


    // This generate a random boolean when called...
    eoBooleanGenerator flip( proba );

    // ... by this initializator...
    eoInitFixedLength<Bools> init( dim, flip );

    // ... when building this pop.
    std::cout << "Init a pop of eoBit of size " << popsize << " at random with p(x=1)=" << proba << std::endl;
    eoPop<Bools> pop( popsize, init );
    // print( pop, "\n" );

    // EDO triplet
    edoBinomial<Bools> distrib(dim,0.0);
    edoEstimatorBinomial<Bools> estimate;
    edoSamplerBinomial<Bools> sample;

    std::cout << "Estimate a distribution from the init pop" << std::endl;
    distrib = estimate(pop);
    print( distrib );
    // std::cout << std::endl;

    std::cout << "Sample a new pop from the init distrib" << std::endl;
    pop.clear();
    for( unsigned int i=0; i<popsize; ++i ) {
        pop.push_back( sample( distrib ) );
    }
    // print( pop, "\n" );

    std::cout << "Estimate a distribution from the sampled pop" << std::endl;
    distrib = estimate(pop);
    print( distrib );
    // std::cout << std::endl;

    double sum=0;
    for( unsigned int i=0; i<distrib.size(); ++i ) {
        sum += distrib[i];
    }
    double e = sum / dim;

    std::cout << "Estimated initial proba = " << e << std::endl;
}
