// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*
(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Caner Candan <caner.candan@thalesgroup.com>

*/

//-----------------------------------------------------------------------------
// t-openmp.cpp
//-----------------------------------------------------------------------------

#include <fstream>
#include <sstream>
#include <climits>

#include <paradiseo/eo.h>
#include <paradiseo/eo/es/make_real.h>

#include <paradiseo/eo/apply.h>

#include <omp.h>

#include <unistd.h>

#include "real_value.h"

//-----------------------------------------------------------------------------

typedef eoReal< eoMinimizingFitness > EOT;

//-----------------------------------------------------------------------------

inline uint32_t get_rdtsc() { __asm__ ("xor %eax, %eax; cpuid; rdtsc"); }

double variable_time_function(const std::vector<double>&)
{
    eoRng myrng( get_rdtsc() );
    ::usleep( myrng.random( 10 ) );
    return 0.0;
}

double measure_apply( size_t p,
		      void (*fct)(eoUF<EOT&, void>&, std::vector<EOT>&),
		      eoInitFixedLength< EOT >& init,
		      eoEvalFuncCounter< EOT >& eval )
{
    eoPop< EOT > pop( p, init );
    double t1 = omp_get_wtime();
    fct( eval, pop );
    double t2 = omp_get_wtime();
    return t2 - t1;
}

void measure( size_t p,
	      eoInitFixedLength< EOT >& init,
	      eoEvalFuncCounter< EOT >& eval,
	      std::ofstream& speedupFile,
	      std::ofstream& efficiencyFile,
	      std::ofstream& dynamicityFile,
	      size_t nbtask )
{
    // sequential scope
    double Ts = measure_apply( p, apply< EOT >, init, eval );
    // parallel scope
    double Tp = measure_apply( p, omp_apply< EOT >, init, eval );
    // parallel scope dynamic
    double Tpd = measure_apply( p, omp_dynamic_apply< EOT >, init, eval );

    double speedup = Ts / Tp;

    if ( speedup > nbtask ) { return; }

    double efficiency = speedup / nbtask;

    speedupFile << speedup << ' ';
    efficiencyFile << efficiency << ' ';

    eo::log << eo::debug;
    eo::log << "Ts = " << Ts << std::endl;
    eo::log << "Tp = " << Tp << std::endl;
    eo::log << "S_p = " << speedup << std::endl;
    eo::log << "E_p = " << efficiency << std::endl;

    double dynamicity = Tp / Tpd;

    if ( dynamicity > nbtask ) { return; }

    eo::log << "Tpd = " << Tpd << std::endl;
    eo::log << "D_p = " << dynamicity << std::endl;

    dynamicityFile << dynamicity << ' ';
}


int main(int ac, char** av)
{
    eoParser parser(ac, av);

    unsigned int popMin = parser.getORcreateParam((unsigned int)1, "popMin", "Population Min", 'p', "Evolution Engine").value();
    unsigned int popStep = parser.getORcreateParam((unsigned int)1, "popStep", "Population Step", 0, "Evolution Engine").value();
    unsigned int popMax = parser.getORcreateParam((unsigned int)100, "popMax", "Population Max", 'P', "Evolution Engine").value();

    unsigned int dimMin = parser.getORcreateParam((unsigned int)1, "dimMin", "Dimension Min", 'd', "Evolution Engine").value();
    unsigned int dimStep = parser.getORcreateParam((unsigned int)1, "dimStep", "Dimension Step", 0, "Evolution Engine").value();
    unsigned int dimMax = parser.getORcreateParam((unsigned int)100, "dimMax", "Dimension Max", 'D', "Evolution Engine").value();

    unsigned int nRun = parser.getORcreateParam((unsigned int)100, "nRun", "Number of runs", 'r', "Evolution Engine").value();

    std::string fileNamesPrefix = parser.getORcreateParam(std::string(""), "fileNamesPrefix", "Prefix of all results files name", 'H', "Results").value();

    std::string speedupFileName = parser.getORcreateParam(std::string("speedup"), "speedupFileName", "Speedup file name", 0, "Results").value();
    std::string efficiencyFileName = parser.getORcreateParam(std::string("efficiency"), "efficiencyFileName", "Efficiency file name", 0, "Results").value();
    std::string dynamicityFileName = parser.getORcreateParam(std::string("dynamicity"), "dynamicityFileName", "Dynamicity file name", 0, "Results").value();

    uint32_t seedParam = parser.getORcreateParam((uint32_t)0, "seed", "Random number seed", 0).value();
    if (seedParam == 0) { seedParam = time(0); }

    unsigned int measureConstTime = parser.getORcreateParam((unsigned int)1, "measureConstTime", "Toggle measure of constant time", 'C', "Results").value();
    unsigned int measureVarTime = parser.getORcreateParam((unsigned int)1, "measureVarTime", "Toggle measure of variable time", 'V', "Results").value();

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_help(parser);
    make_verbose(parser);

    rng.reseed( seedParam );

    eoEvalFuncPtr< EOT, double, const std::vector< double >& > mainEval( real_value );
    eoEvalFuncCounter< EOT > eval( mainEval );

    eoEvalFuncPtr< EOT, double, const std::vector< double >& > mainEval_variable( variable_time_function );
    eoEvalFuncCounter< EOT > eval_variable( mainEval_variable );

    eoUniformGenerator< double > gen(-5, 5);

    std::ostringstream params;
    params << "_p" << popMin << "_pS" << popStep << "_P" << popMax
	   << "_d" << dimMin << "_dS" << dimStep << "_D" << dimMax
	   << "_r" << nRun << "_s" << seedParam;

    std::ofstream speedupFile( std::string( fileNamesPrefix + speedupFileName + params.str() ).c_str() );
    std::ofstream efficiencyFile( std::string( fileNamesPrefix + efficiencyFileName + params.str() ).c_str() );
    std::ofstream dynamicityFile( std::string( fileNamesPrefix + dynamicityFileName + params.str() ).c_str() );

    std::ofstream speedupFile_variable( std::string( fileNamesPrefix + "variable_" + speedupFileName + params.str() ).c_str() );
    std::ofstream efficiencyFile_variable( std::string( fileNamesPrefix + "variable_" + efficiencyFileName + params.str() ).c_str() );
    std::ofstream dynamicityFile_variable( std::string( fileNamesPrefix + "variable_" + dynamicityFileName + params.str() ).c_str() );

    size_t nbtask = 1;
#pragma omp parallel
    {
	nbtask = omp_get_num_threads();
    }

    eo::log << eo::logging << "Nb task: " << nbtask << std::endl;

    for ( size_t p = popMin; p <= popMax; p += popStep )
	{
	    for ( size_t d = dimMin; d <= dimMax; d += dimStep )
	    	{
		    eo::log << eo::logging << p << 'x' << d << std::endl;

		    for ( size_t r = 0; r < nRun; ++r )
			{
			    eoInitFixedLength< EOT > init( d, gen );

			    // for constant time measure
			    if ( measureConstTime == 1 )
				{
				    measure( p, init, eval, speedupFile, efficiencyFile, dynamicityFile, nbtask );
				}

			    // for variable time measure
			    if ( measureVarTime == 1 )
				{
				    measure( p, init, eval_variable, speedupFile_variable, efficiencyFile_variable, dynamicityFile_variable, nbtask );
				}
			} // end of runs

		    if ( measureConstTime == 1 )
			{
			    speedupFile << std::endl;
			    efficiencyFile << std::endl;
			    dynamicityFile << std::endl;
			}

		    if ( measureVarTime == 1 )
			{
			    speedupFile_variable << std::endl;
			    efficiencyFile_variable << std::endl;
			    dynamicityFile_variable << std::endl;
			}

		} // end of dimension

	} // end of population

    return 0;
}

//-----------------------------------------------------------------------------
