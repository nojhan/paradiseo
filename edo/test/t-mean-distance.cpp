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

#include <sys/stat.h>
#include <sys/types.h>

#include <sstream>
#include <iomanip>
#include <fstream>

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

    unsigned int r_max = parser.createParam((unsigned int)10, "run-number", "Number of run", 'r', section).value(); // r
    unsigned int p_min = parser.createParam((unsigned int)10, "population-min", "Population min", 'p', section).value(); // p
    unsigned int p_max = parser.createParam((unsigned int)100, "population-max", "Population max", 'P', section).value(); // P
    unsigned int p_step = parser.createParam((unsigned int)50, "population-step", "Population step", 't', section).value(); // t
    unsigned int s_size = parser.createParam((unsigned int)2, "dimension-size", "Dimension size", 'd', section).value(); // d

    AtomType mean_value = parser.createParam((AtomType)0, "mean", "Mean value", 'm', section).value(); // m
    AtomType covar1_value = parser.createParam((AtomType)1.0, "covar1", "Covar value 1", '1', section).value(); // 1
    AtomType covar2_value = parser.createParam((AtomType)0.5, "covar2", "Covar value 2", '2', section).value(); // 2
    AtomType covar3_value = parser.createParam((AtomType)1.0, "covar3", "Covar value 3", '3', section).value(); // 3

    std::string results_directory = parser.createParam((std::string)"means_distances_results", "results-directory", "Results directory", 'R', section).value(); // R
    std::string files_description = parser.createParam((std::string)"files_description.txt", "files-description", "Files description", 'F', section).value(); // F

    if (parser.userNeedsHelp())
        {
            parser.printHelp(std::cout);
            exit(1);
        }

    make_verbose(parser);
    make_help(parser);



    assert(r_max >= 1);
    assert(s_size >= 2);

    eo::log << eo::quiet;

    ::mkdir( results_directory.c_str(), 0755 );

    for ( unsigned int p_size = p_min; p_size <= p_max; p_size += p_step )
        {
            assert(p_size >= p_min);

            std::ostringstream desc_file;
            desc_file << results_directory << "/" << files_description;

            std::ostringstream cur_file;
            cur_file << results_directory << "/pop_" << p_size << ".txt";

            eo::log << eo::file( desc_file.str() ) << cur_file.str().c_str() << std::endl;

            eo::log << eo::file( cur_file.str() );

            eo::log << eo::logging << "run_number p_size s_size mean(0) mean(1) new-mean(0) new-mean(1) distance" << std::endl;

            eo::log << eo::quiet;

            for ( unsigned int r = 1; r <= r_max; ++r)
                {

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


#ifdef WITH_BOOST
                    Vector mean( s_size, mean_value );
#else
#ifdef WITH_EIGEN
                    Vector mean( s_size );
                    mean = Vector::Constant( s_size, mean_value);
#endif
#endif
                    Matrix varcovar( s_size, s_size );

                    varcovar( 0, 0 ) = covar1_value;
                    varcovar( 0, 1 ) = covar2_value;
                    varcovar( 1, 1 ) = covar3_value;

                    Distrib distrib( mean, varcovar );





                    // Prepare bounder class to set bounds of sampling.
                    // This is used by edoSampler.


                    edoBounder< EOT >* bounder = new edoBounderRng< EOT >(EOT(pop[0].size(), -5),
                                                                        EOT(pop[0].size(), 5),
                                                                        *gen);
                    state.storeFunctor(bounder);





                    // Prepare sampler class with a specific distribution


                    edoSampler< Distrib >* sampler = new edoSamplerNormalMulti< EOT >( *bounder );
                    state.storeFunctor(sampler);





                    // (4) sampling phase


                    pop.clear();

                    for (unsigned int i = 0; i < p_size; ++i)
                        {
                            EOT candidate_solution = (*sampler)( distrib );
                            pop.push_back( candidate_solution );
                        }





                    // (6) estimation phase


                    edoEstimator< Distrib >* estimator = new edoEstimatorNormalMulti< EOT >();
                    state.storeFunctor(estimator);

                    distrib = (*estimator)( pop );





                    // (8) euclidianne distance estimation


                    Vector new_mean = distrib.mean();
                    Matrix new_varcovar = distrib.varcovar();

                    AtomType distance = 0;

                    for ( unsigned int d = 0; d < s_size; ++d )
                        {
                            distance += pow( mean[ d ] - new_mean[ d ], 2 );
                        }

                    distance = sqrt( distance );

                    eo::log << r << " " << p_size << " " << s_size << " "
                            << mean(0) << " " << mean(1) << " "
                            << new_mean(0) << " " << new_mean(1) << " "
                            << distance << std::endl
                        ;



                }

        }

    return 0;
}
