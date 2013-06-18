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

#ifndef _moTrikiCoolingSchedule_h
#define _moTrikiCoolingSchedule_h

#include <coolingSchedule/moCoolingSchedule.h>

#include <continuator/moNeighborhoodStat.h>
#include <continuator/moStdFitnessNeighborStat.h>
#include <neighborhood/moNeighborhood.h>
#include <continuator/moStat.h>
#include <continuator/moFitnessMomentsStat.h>


/*
#include <continuator/moStat.h>

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>
 */

#include <iostream>
using namespace std;


//!
/*!
 */
template< class EOT, class Neighbor > //, class Neighborhood >
class moTrikiCoolingSchedule: public moCoolingSchedule< EOT >
{
public:
	//typedef typename Neighbor::EOT EOT ;
	typedef moNeighborhood<Neighbor> Neighborhood ;

	//! Constructor
	/*!
	 */

	moTrikiCoolingSchedule (Neighborhood& _neighborhood, moEval<Neighbor>& _eval, double _initTemp)
	: initTemp(_initTemp),
	  mu2(10),                   // mu2 typically belongs to [1; 20]
	  K1(2),                    // K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
	  K2(5), // ???
	  lambda1(2), // the increase in temperature, typically in [1.5; 4]
	  lambda2(.7), // lambda2 in [0.5; 0.99]
	  mu1(10), // target decrease in cost factor, in [2; 20]
	  xi(1.05), // xi typically belongs to [1; 1.1]
	  max_accepted(50),   // depends on pb/neighborhood
	  max_generated(100), // depends on pb/neighborhood
	  theta(10), // theta is typically set to 10
	  statIsInitialized(false),
	  outf("out.data")
	{ }
	
	moTrikiCoolingSchedule (
			Neighborhood& _neighborhood, moEval<Neighbor>& _eval, double _initTemp,
			double _max_accepted,
			double _max_generated
	)
	: initTemp(_initTemp),
	  mu2(10),                   // mu2 typically belongs to [1; 20]
	  K1(2),                    // K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
	  K2(5), // ???
	  lambda1(2), // the increase in temperature, typically in [1.5; 4]
	  lambda2(.7), // lambda2 in [0.5; 0.99]
	  mu1(10), // target decrease in cost factor, in [2; 20]
	  xi(1.05), // xi typically belongs to [1; 1.1]
	  max_accepted(_max_accepted),   // depends on pb/neighborhood
	  max_generated(_max_generated), // depends on pb/neighborhood
	  theta(10), // theta is typically set to 10
	  statIsInitialized(false),
	  outf("out.data")
	{ }
	
	/**
	 * Initial temperature
	 * @param _solution initial solution
	 */
	double init(EOT & _solution) {
		
		accepted = generated = costs_sum = 0;
		
		negative_temp = equilibrium_not_reached = frozen = 0;
		
		reinitializing = false;
		terminated = false;
		statIsInitialized = false;
		
		///
		cout << "INIT T=" << initTemp << endl;
		///
		
		//outf.open("out");
		//outf << "ok";
		//outf.close();
		
		
		return initTemp;
	}

	/**
	 * update the temperature by a factor
	 * @param _temp current temperature to update
	 * @param _acceptedMove true when the move is accepted, false otherwise
	 */
	void update(double& _temp, bool _acceptedMove, EOT & _solution) {
		
		//cout << _temp << "  g " << generated << endl;
		
		generated++;
		
		if (_acceptedMove)
		{
			accepted++;
			//costs_sum += _solution.fitness();
			//varStat(_solution);
			if (statIsInitialized)
				 momStat(_solution);
			else momStat.init(_solution), statIsInitialized = true;
			
			//cout << _solution.fitness() << "  avgCost=" << momStat.value().first << endl;
		}
		
		
		if (accepted > max_accepted || generated > max_generated) {
			
			if (accepted == 0) // ADDED! Otherwise the computed std dev is null; we're probably at equilibrium
			{
				///
				cout << "Stopping: no accepted solution" << endl;
				///
				
				terminated = true;
				return;
			}
			
			///
			cout << (accepted > max_accepted? "MAXACC  ": "MAXGEN  ");
			///
			
			//double avgCost = costs_sum/(double)accepted;
			//double stdDev = sqrt(varStat.value()); // WARNING: IT'S NO MORE THE AVG COST, NOW IT'S THE STD DEV!
			//double variance = varStat.value();
			double avgCost = momStat.value().first;
			double variance = momStat.value().second;
			double stdDev = sqrt(variance);
			double sigma = stdDev;
			double delta = sigma/mu2;
			
			
			//outf << avgCost << endl;
			//outf << _temp << endl;
			outf << prevAvgCost-delta << endl;
			
			
			accepted = generated = costs_sum = 0;
			//varStat.init(_solution);//TODON use next chain's first sol
			//momStat.init(_solution);//TODONE use next chain's first sol
			statIsInitialized = false;
			
			///
			cout << "T=" << _temp << "  avgCost=" << avgCost << "  stdDev=" << stdDev << "  currCost=" << _solution.fitness() << endl;
			///
			
			double alpha;
			double oldprevAvgCost = prevAvgCost;
			
			///
			cout << "negTemp: " << negative_temp << " / " << K2 << endl;
			///
			
			if (negative_temp < K2)
			{
				if (!reinitializing)
				{
					///
					if (avgCost/(prevAvgCost-delta) > xi) cout << "/!\\ eq not reached!" << endl;
					///
					
					if (avgCost/(prevAvgCost-delta) > xi)
						 equilibrium_not_reached++;
					else equilibrium_not_reached = 0;
				}
				if (equilibrium_not_reached > K1)
				{
					///
					cout << "/!\\ Reinitializing (eq not reached)" << endl;
					///
					
					reinitializing = true;
					alpha = lambda1;
					delta = sigma/mu1;
					equilibrium_not_reached = 0; // ADDED! Otherwise the algo gets trapped here!
				}
				else if (_temp*delta/(sigma*sigma) >= 1)
				{
					///
					cout << "/!\\ neg temp!" << endl;
					///
					
					negative_temp++;
					reinitializing = true;
					if (negative_temp < K2)
					{
						alpha = lambda1;
						delta = sigma/mu1;
					} else
						alpha = lambda2;
				}
			
			// First interpretation of the pseudocode indentation: (seems obviously false because it makes the above code unreachable)
			/*
			}
			else
			{
				cout << "ccc" << endl;
				reinitializing = false;
				prevAvgCost = avgCost;
				alpha = 1-_temp*delta/(sigma*sigma);
			}
			*/
				
				// Second interpretation of the pseudocode indentation:
				else
				{
					///
					cout << "[normal decrease]" << endl;
					///
					
					reinitializing = false;
					prevAvgCost = avgCost;
					//alpha = 1-_temp*delta/(sigma*sigma);
					alpha = 1-_temp*delta/variance;
					
					//alpha = (sigma==0? 1: 1-_temp*delta/(sigma*sigma)); // ADDED! but removed
					
					if (sigma == 0) // ADDED! When std dev is null, the solution is probably at eq, and the algo can't go on anyways
						terminated = true, cout << "Stopping: null std dev" << endl;
				}
			}
			// FIXME: else what? alpha=?
			
			

			///
			cout << "*=" << alpha << endl;
			///
			
			_temp *= alpha;
			
			
			// Never seems to be used
			if (avgCost == oldprevAvgCost) // use a neighborhood to approximate double equality?
				 frozen++;
			else frozen = 0;
			
			
			
			//exit(0);
			//cin.get();
			
			
		}
		
	}

	//! Function which proceeds to the cooling
	/*!
	 */
	bool operator() (double temperature)
	{
		///
		if (terminated) cout << "TERMINATED" << endl;
		///
		
		return frozen < theta
				&& !terminated ; // ADDED! because 'frozen' doesn't terminate anything
	}

private:
	//moNeighborhoodStat<Neighbor> nhStat;
	//moStdFitnessNeighborStat<Neighbor> stdDevStat;
	const double
	// parameters of the algorithm
		//currentTemp,
		initTemp,
		//ratio,
		//threshold,
		mu2,                   // mu2 typically belongs to [1; 20]
		K1,                    // K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
		K2,
		lambda1, // the increase in temperature, typically in [1.5; 4]
		lambda2, // lambda2 in [0.5; 0.99]
		mu1, // target decrease in cost factor, in [2; 20]
		xi // xi typically belongs to [1; 1.1]
	// private variables
	;
	double
		stdDev,
		prevAvgCost,
		expectedDecreaseInCost, // delta
		costs_sum
	;
	const int
		max_accepted,
		max_generated,
		theta // theta is typically set to 10
	;
	int
		accepted,
		generated,
		equilibrium_not_reached,
		negative_temp,
		frozen
	;
	bool reinitializing, terminated;
	
	//moFitnessVarianceStat<EOT> varStat;
	moFitnessMomentsStat<EOT> momStat;
	bool statIsInitialized;
	
	ofstream outf;
	
};

#endif

