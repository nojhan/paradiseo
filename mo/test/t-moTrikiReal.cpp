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
Lionel Parreaux <lionel.parreaux@gmail.com>

*/

#include <mo>
#include <eo>
#include <es.h>
#include <edo>

//#include <trikisa>

#include <edoBounderUniform.h>

#include "neighborhood/moRealNeighbor.h"
#include "neighborhood/moRealNeighborhood.h"

#include "sampling/moStdDevEstimator.h"

#include "coolingSchedule/moTrikiCoolingSchedule.h"


//#include "moRealInitTemperature.h"

//#include "do/make_real_init_temperature.h"

typedef eoReal< eoMinimizingFitness > EOT;
typedef moRealNeighbor< EOT > Neighbor;

double objective_function(const EOT & sol)
{
    double sum = 0;

    for ( size_t i = 0; i < sol.size(); ++i )
	{
	    sum += sol[i] * sol[i];
	}

    return sum;
}

int main( int ac, char** av )
{
    eoParser parser( ac, av );

    eoState state;

    eoEvalFuncPtr< EOT, double > eval( objective_function );
    moFullEvalByCopy< Neighbor > fullEval( eval );
    
    //moNeighborhood< Neighbor >* neighborhood;
    
    int dimSize = 20;
    
    //moRealInitTemperature< EOT >& real_init = do_make_real_init_temperature( parser, state, eval );
    //moInitTemperature< EOT, Neighbor >& real_init = do_make_real_init_temperature( parser, state, eval, neval );
    //moInitTemperature< EOT, Neighbor >& real_init = do_make_init_temperature<EOT, Neighbor>( parser, state, eval, neval, neighborhood );
    //moStdDevEstimator< EOT, Neighbor >& real_init = do_make_init_temperature<EOT, Neighbor>( parser, state, eval, neval, neighborhood, dimSize );
    

    //-------------------------------------------------------
    // Parameters
    //-------------------------------------------------------

    std::string section( "Temperature initialization paramaters" );

    unsigned int dimension_size = parser.getORcreateParam( (unsigned int)dimSize, "dimension-size", "Dimension size", 'd', section ).value();
    double jump_bound = parser.getORcreateParam( (double)1, "jump-bound", "Bound of jump", '\0', section ).value();
    unsigned int maxiter = parser.getORcreateParam( (unsigned int)10, "max-iter", "Maximum number of iterations", '\0', section ).value();
    //unsigned int ratio = parser.getORcreateParam( (unsigned int)1, "ratio", "Bounder ratio", '\0', section ).value(); // not used

    //-------------------------------------------------------


    //-------------------------------------------------------
    // Instanciate needed classes
    //-------------------------------------------------------

    edoUniform< EOT > distrib( EOT(dimension_size, -1 * jump_bound), EOT(dimension_size, 1 * jump_bound) );
    
    edoBounder< EOT > bounder_search( EOT(dimension_size, -10), EOT(dimension_size, 10) );
    
    edoSamplerUniform< EOT > sampler( bounder_search );
    
    //moRealNeighborhood< edoUniform< EOT >, Neighbor >* neighborhood = new moRealNeighborhood< edoUniform< EOT >, Neighbor >( *distrib, *sampler, *bounder_search );
    moRealNeighborhood< edoUniform< EOT >, Neighbor > neighborhood( distrib, sampler, bounder_search );
    //state.storeFunctor(neighborhood);state.storeFunctor(neighborhood);

    //moStdDevEstimator< EOT, Neighbor >* init = new moStdDevEstimator< EOT, Neighbor >( *neighborhood, fullEval, eval, maxiter );
    //moStdDevEstimator< EOT, Neighbor > init( maxiter, neighborhood, fullEval, eval );
    moStdDevEstimator< EOT, Neighbor > init( maxiter, neighborhood, fullEval );
    
    //-------------------------------------------------------



    
    //-------------------------------------------------------
    // Help + Verbose routines
    //-------------------------------------------------------

    if (parser.userNeedsHelp())
        {
            parser.printHelp(std::cout);
            exit(1);
        }

    make_help(parser);

    //-------------------------------------------------------


    //EOT solution(2, 5);
    EOT solution(dimSize, 5);
    /*
    real_init( solution );

    std::cout << "do_make_real_init_temperature( parser, eval ): "
	      << real_init.value()
	      << std::endl;
	*/

    std::cout << "do_make_real_init_temperature( parser, eval ): "
	      << init( solution )
	      << std::endl;
    
    
    

	moTrueContinuator<Neighbor> continuator;
	moCheckpoint<Neighbor> checkpoint(continuator);
	moFitnessStat<EOT> fitStat;
	checkpoint.add(fitStat);
	eoFileMonitor monitor("triki.out", "");
	//eoGnuplot1DMonitor monitor2("trikignu.out", true);
	moCounterMonitorSaver countMon(1, monitor);
	checkpoint.add(countMon);
	//moCounterMonitorSaver gnuMon (1, monitor2);
	//checkpoint.add(gnuMon);
	monitor.add(fitStat);
	//monitor2.add(fitStat);
	
    
    
	moTrikiCoolingSchedule<EOT, Neighbor> coolingSchedule (
			neighborhood, neval, init( solution ),
			//50,
			200, //150,
			//100
			350 // 250
		);
	moSA<Neighbor> localSearch(*neighborhood, eval, fullEval, coolingSchedule, checkpoint);
	
	std::cout << "#########################################" << std::endl;
	std::cout << "initial solution1: " << solution << std::endl ;
	
	localSearch(solution);
	
	std::cout << "final solution1: " << solution << std::endl ;
	std::cout << "#########################################" << std::endl;
	
	//delete neighborhood;
    
    return 0;
}



