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

#include <sstream>
#include <iomanip>

#include <eo>
//#include <mo>

#include <edo>

#include "../application/common/Rosenbrock.h"
#include "../application/common/Sphere.h"

typedef eoReal< eoMinimizingFitness > EOT;
typedef edoNormalMulti< EOT > Distrib;
typedef EOT::AtomType AtomType;

#ifdef WITH_BOOST
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
    typedef ublas::vector< AtomType > Vector;
    typedef ublas::symmetric_matrix< AtomType, ublas::lower > Matrix;
#else
#ifdef WITH_EIGEN
#include <Eigen/Dense>
    typedef edoNormalMulti<EOT>::Vector Vector;
    typedef edoNormalMulti<EOT>::Matrix Matrix;
#endif
#endif

int main(int ac, char** av)
{
    // (0) parser + eo routines
    eoParser parser(ac, av);

    std::string        section("Algorithm parameters");

    unsigned int p_size   = parser.createParam((unsigned int)100, "popSize", "Population Size", 'P', section).value(); // P
    unsigned int s_size   = parser.createParam((unsigned int)2, "dimension-size", "Dimension size", 'd', section).value(); // d
    AtomType mean_value   = parser.createParam((AtomType)0, "mean", "Mean value", 'm', section).value(); // m
    AtomType covar1_value = parser.createParam((AtomType)1.0, "covar1", "Covar value 1", '1', section).value();
    AtomType covar2_value = parser.createParam((AtomType)0.5, "covar2", "Covar value 2", '2', section).value();
    AtomType covar3_value = parser.createParam((AtomType)1.0, "covar3", "Covar value 3", '3', section).value();

    std::ostringstream ss;
    ss << p_size << "_" << std::fixed << std::setprecision(1)
       << mean_value << "_" << covar1_value << "_" << covar2_value << "_"
       << covar3_value << "_gen";
    std::string gen_filename = ss.str();

    if( parser.userNeedsHelp() ) {
            parser.printHelp(std::cout);
            exit(1);
    }

    make_verbose(parser);
    make_help(parser);

    assert(p_size > 0);
    assert(s_size > 0);

    eoState state;

    // (1) Population init and sampler
    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
    state.storeFunctor(gen);

    eoInitFixedLength< EOT >* init = new eoInitFixedLength< EOT >( s_size, *gen );
    state.storeFunctor(init);

    // create an empty pop and let the state handle the memory
    // fill population thanks to eoInit instance
    eoPop< EOT >& pop = state.takeOwnership( eoPop< EOT >( p_size, *init ) );

    // (2) distribution initial parameters
    Vector mean( s_size );

    for (unsigned int i = 0; i < s_size; ++i) {
        mean( i ) = mean_value; 
    }

    Matrix varcovar( s_size, s_size );

    varcovar( 0, 0 ) = covar1_value;
    varcovar( 0, 1 ) = covar2_value;
    varcovar( 1, 1 ) = covar3_value;

    Distrib distrib( mean, varcovar );

    // (3a) distribution output preparation
    edoDummyContinue< Distrib >* distrib_dummy_continue = new edoDummyContinue< Distrib >();
    state.storeFunctor(distrib_dummy_continue);

    edoCheckPoint< Distrib >* distrib_continue = new edoCheckPoint< Distrib >( *distrib_dummy_continue );
    state.storeFunctor(distrib_continue);

    edoDistribStat< Distrib >* distrib_stat = new edoStatNormalMulti< EOT >();
    state.storeFunctor(distrib_stat);

    distrib_continue->add( *distrib_stat );

    edoFileSnapshot* distrib_file_snapshot = new edoFileSnapshot( "TestResDistrib", 1, gen_filename );
    state.storeFunctor(distrib_file_snapshot);
    distrib_file_snapshot->add(*distrib_stat);
    distrib_continue->add(*distrib_file_snapshot);

    // (3b) distribution output
    (*distrib_continue)( distrib );

    // Prepare bounder class to set bounds of sampling.
    // This is used by edoSampler.
    edoBounder< EOT >* bounder = new edoBounderRng< EOT >(
            EOT(pop[0].size(), -5), EOT(pop[0].size(), 5), *gen
        );
    state.storeFunctor(bounder);

    // Prepare sampler class with a specific distribution
    edoSampler< Distrib >* sampler = new edoSamplerNormalMulti< EOT >( *bounder );
    state.storeFunctor(sampler);

    // (4) sampling phase
    pop.clear();

    for( unsigned int i = 0; i < p_size; ++i ) {
        EOT candidate_solution = (*sampler)( distrib );
        pop.push_back( candidate_solution );
    }

    // (5) population output
    eoContinue< EOT >* pop_cont = new eoGenContinue< EOT >( 2 ); // never reached fitness
    state.storeFunctor(pop_cont);

    eoCheckPoint< EOT >* pop_continue = new eoCheckPoint< EOT >( *pop_cont );
    state.storeFunctor(pop_continue);

    edoPopStat< EOT >* pop_stat = new edoPopStat<EOT>;
    state.storeFunctor(pop_stat);
    pop_continue->add(*pop_stat);

    edoFileSnapshot* pop_file_snapshot = new edoFileSnapshot( "TestResPop", 1, gen_filename );
    state.storeFunctor(pop_file_snapshot);
    pop_file_snapshot->add(*pop_stat);
    pop_continue->add(*pop_file_snapshot);

    (*pop_continue)( pop );

    // (6) estimation phase
    edoEstimator< Distrib >* estimator = new edoEstimatorNormalMulti< EOT >();
    state.storeFunctor(estimator);

    distrib = (*estimator)( pop );

    // (7) distribution output
    (*distrib_continue)( distrib );

    // (8) euclidianne distance estimation
    Vector new_mean = distrib.mean();
    Matrix new_varcovar = distrib.varcovar();

    AtomType distance = 0;
    for( unsigned int d = 0; d < s_size; ++d ) {
        distance += pow( mean[ d ] - new_mean[ d ], 2 );
    }

    distance = sqrt( distance );

    eo::log << eo::logging
            << "mean: " << mean << std::endl
            << "new mean: " << new_mean << std::endl
            << "distance: " << distance << std::endl
        ;

    return 0;
}
