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

/*
static const char * stoppingReasons[] = {"no accepted solution", "null std dev" , "frozen >= theta"};
static const char * chainEndingReasons[] = {"MAX GENerated solutions", "MAX ACCepted solutions"};
*/

//!
/*!
 */
//template< class Neighbor > //, class Neighborhood >
//class moTrikiCoolingSchedule: public moCoolingSchedule< typename Neighbor::EOT >
template< class EOT >
class moTrikiCoolingSchedule: public moCoolingSchedule< EOT >
{
public:
    //typedef typename Neighbor::EOT EOT ;
    //typedef moNeighborhood<Neighbor> Neighborhood ;

    //! Constructor
    /*!
     */
    /*
    moTrikiCoolingSchedule (Neighborhood& _neighborhood, moEval<Neighbor>& _eval, double _initTemp) // FIXME rem useless params!!
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
      statIsInitialized(false)//,
      //outf("out.data")
    { }
*/
    
//    moTrikiCoolingSchedule (
//            Neighborhood& _neighborhood, moEval<Neighbor>& _eval, double _initTemp,
//            double _max_accepted,
//            double _max_generated
//    )
//    : initTemp(_initTemp),
//      mu2(10),                   // mu2 typically belongs to [1; 20]
//      K1(2),                    // K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
//      K2(5), // ???
//      lambda1(2), // the increase in temperature, typically in [1.5; 4]
//      lambda2(.7), // lambda2 in [0.5; 0.99]
//      mu1(10), // target decrease in cost factor, in [2; 20]
//      xi(1.05), // xi typically belongs to [1; 1.1]
//      max_accepted(_max_accepted),   // depends on pb/neighborhood
//      max_generated(_max_generated), // depends on pb/neighborhood
//      theta(10), // theta is typically set to 10
//      statIsInitialized(false)//,
//      //outf("out.data")
//    { }
    
    moTrikiCoolingSchedule (
        double _initTemp,
        int _max_accepted = 50,
        int _max_generated = 100,
        double _mu2 = 2,
        double _mu1 = 10,
        double _lambda1 = 2,
        double _lambda2 = .7,
        double _xi = 1.05,
        int _theta = 10,
        int _K1 = 10,
        int _K2 = 5
    ) // TODO reorder inits
    : initTemp(_initTemp),
      mu2(_mu2),                   // mu2 typically belongs to [1; 20]
      K1(_K1),                    // K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
      K2(_K2), // ???
      lambda1(_lambda1), // the increase in temperature (reheating factor), typically in [1.5; 4]
      lambda2(_lambda2), // lambda2 in [0.5; 0.99]
      mu1(_mu1), // target decrease in cost factor, in [2; 20]
      xi(_xi), // xi typically belongs to [1; 1.1]
      max_accepted(_max_accepted),   // depends on pb/neighborhood
      max_generated(_max_generated), // depends on pb/neighborhood
      theta(_theta), // theta is typically set to 10
      statIsInitialized(false)
    {
        chainStat.temperature = initTemp;
    }
    
    /**
     * Initial temperature
     * @param _solution initial solution
     */
    double init(EOT & _solution) {
        
        accepted = generated = 0;// = costs_sum = 0;
        
        negative_temp = equilibrium_not_reached = frozen = 0;
        
        chainStat.delta = initTemp/mu2;
        
        //cout << "acc " << max_accepted << " " << max_generated << endl;
        
        /*
        reinitializing = false;
        terminated = false;
        statIsInitialized = false;
        */
        reinitializing = false;
        //cout << "INIT T=" << initTemp << endl;
        //cout << "INIT T=" << chainStat.temperature << endl;
        
        //chainStat.temperature = initTemp;
                
        ///
        //cout << "INIT T=" << initTemp << endl;
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
        chainStat.temperature = _temp;
        chainStat.stoppingReason = NULL;
        chainStat.chainEndingReason = NULL;
        chainStat.equilibriumNotReached = false;
        chainStat.negativeTemp = false;
        //chainStat.markovChainEnded = false;
        chainStat.generatedSolutions = generated;
        chainStat.acceptedSolutions = accepted;
        
        generated++;
        //cout << "gen " << generated << endl;
        
        
        if (_acceptedMove)
        {
            accepted++;
            //costs_sum += _solution.fitness();
            //varStat(_solution);
            if (statIsInitialized)
                 momentStat(_solution);
            else momentStat.init(_solution), statIsInitialized = true;
            
            //cout << _solution.fitness() << "  avgFitness=" << momentStat.value().first << endl;
        }
        
        if (accepted > max_accepted || generated > max_generated) {

            chainStat.chainEndingReason = accepted > max_accepted ? chainEndingReasons[0]: chainEndingReasons[1];
            
            //chainStat.markovChainEnded = true;
            
            if (accepted == 0) // ADDED! Otherwise the computed std dev is null; we're probably at equilibrium
            {
                ///
                //cout << "Stopping: no accepted solution" << endl;
                ///
                
                chainStat.terminated = true, chainStat.stoppingReason = stoppingReasons[0];
                return; // FIXME nutgud
            }
            
            ///
            //cout << (accepted > max_accepted? "MAXACC  ": "MAXGEN  ");
            ///
            
            //double avgFitness = costs_sum/(double)accepted;
            //double stdDev = sqrt(varStat.value()); // WARNING: IT'S NO MORE THE AVG COST, NOW IT'S THE STD DEV!
            //double variance = varStat.value();
            double avgFitness = momentStat.value().first;
            double variance = momentStat.value().second;
            //double stdDev = sqrt(variance);
            chainStat.stdDev = sqrt(variance);
            double sigma = chainStat.stdDev;
            ///////////////
            //double delta = sigma/mu2;
            ///////////////
            
            
            //outf << avgFitness << endl;
            //outf << _temp << endl;
            //outf << prevAvgFitness-delta << endl;
            
            
            accepted = generated = 0; // = costs_sum = 0;
            //momentStat.init(_solution);//TODONE use next chain's first sol
            statIsInitialized = false;
            
            ///
            //cout << "T=" << _temp << "  avgFitness=" << avgFitness << "  stdDev=" << chainStat.stdDev << "  currCost=" << _solution.fitness() << endl;
            ///
            
            double alpha = 0;
            double prevAvgFitness = chainStat.avgFitness;
            
            ///
            //cout << "negTemp: " << negative_temp << " / " << K2 << endl;
            ///
            
            if (negative_temp < K2)
            {
                //if (!chainStat.reinitializing)
                if (!reinitializing)
                {
                    ///
                    //if (avgFitness/(chainStat.prevAvgFitness-delta) > xi) cout << "/!\\ eq not reached!" << endl;
                    ///
                    
                    if (avgFitness/(prevAvgFitness-chainStat.delta) > xi)
                         equilibrium_not_reached++, chainStat.equilibriumNotReached = true;
                    else equilibrium_not_reached = 0;
                }
                if (equilibrium_not_reached > K1)
                {
                    ///
                    //cout << "/!\\ Reinitializing (eq not reached)" << endl;
                    ///

                    //chainStat.reinitializing = true;
                    reinitializing = true;
                    //chainStat.equilibriumNotReached = true;
                    
                    alpha = lambda1;
                    chainStat.delta = sigma/mu1;
                    equilibrium_not_reached = 0; // ADDED! Otherwise the algo gets trapped here!
                }
                else if (_temp*chainStat.delta/(sigma*sigma) >= 1)
                {
                    ///
                    //cout << "/!\\ neg temp!" << endl;
                    ///
                    
                    negative_temp++;
                    reinitializing = true;
                    chainStat.negativeTemp = true;
                    
                    if (negative_temp < K2)
                    {
                        alpha = lambda1;
                        chainStat.delta = sigma/mu1;
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
                prevAvgFitness = avgFitness;
                alpha = 1-_temp*delta/(sigma*sigma);
            }
            */
                
                // Second interpretation of the pseudocode indentation:
                else
                {
                    ///
                    //cout << "[normal decrease]" << endl;
                    ///
                    
                    reinitializing = false;
                    //chainStat.avgFitness = avgFitness;
                    //alpha = 1-_temp*delta/(sigma*sigma);
                    alpha = 1-_temp*chainStat.delta/variance;
                    
                    //alpha = (sigma==0? 1: 1-_temp*delta/(sigma*sigma)); // ADDED! but removed
                    
                    if (sigma == 0) // ADDED! When std dev is null, the solution is probably at eq, and the algo can't go on anyways
                        chainStat.terminated = true, chainStat.stoppingReason = stoppingReasons[1]; //, cout << "Stopping: null std dev" << endl;
                }
            }
            // FIXME: else what? alpha=?
            
            ///
            //cout << "*=" << alpha << endl;
            ///
            
            _temp *= alpha;

            chainStat.currentFitness = _solution.fitness(); // FIXME here?
            chainStat.alpha = alpha;
            //chainStat.delta = delta;
            chainStat.avgFitness = avgFitness;
            
            
            // Never seems to be used
            if (avgFitness == prevAvgFitness) // use a neighborhood to approximate double floating equality?
                 frozen++;
            else frozen = 0;
            
            
            if (frozen >= theta)
                    chainStat.terminated = true, chainStat.stoppingReason = stoppingReasons[2];;
            
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
        //if (chainStat.terminated) cout << "TERMINATED" << endl;
        ///
        
        return frozen < theta
                && !chainStat.terminated ; // ADDED! because 'frozen' doesn't terminate anything
        
        
        return !chainStat.terminated ; // ADDED! because 'frozen' doesn't terminate anything
        
    }
    
    /*
    bool markovChainJustEnded() const
    {
        return markovChainEnded;
    }*/
    
    
    // Code for generating the getters:
/*
#define __triki_getterType double
#define __triki_makeGetter(name) __triki_getterType name() { return chainStat.name; }
    
    __triki_makeGetter(stdDev)
    __triki_makeGetter(avgFitness)
    __triki_makeGetter(prevTemp)
    __triki_makeGetter(currentFitness)
    __triki_makeGetter(alpha)

#undef __triki_getterType
#define __triki_getterType int
    
    __triki_makeGetter(generatedSolutions)
    __triki_makeGetter(acceptedSolutions)

#undef __triki_getterType
#define __triki_getterType const char *
    
    __triki_makeGetter(stoppingReason)
    __triki_makeGetter(chainEndingReason)

#undef __triki_getterType
#undef __triki_makeGetter
    */

#define __triki_makeGetter(name, type) type name() { return chainStat.name; }
    
    __triki_makeGetter(stdDev, double)
    __triki_makeGetter(avgFitness, double)
    __triki_makeGetter(temperature, double)
    __triki_makeGetter(currentFitness, double)
    __triki_makeGetter(alpha, double)
    __triki_makeGetter(delta, double)
    
    __triki_makeGetter(generatedSolutions, int)
    __triki_makeGetter(acceptedSolutions, int)

    __triki_makeGetter(equilibriumNotReached, bool)
    __triki_makeGetter(negativeTemp, bool)
    __triki_makeGetter(terminated, bool)
    
    __triki_makeGetter(stoppingReason, const char *)
    __triki_makeGetter(chainEndingReason, const char *)

#undef __triki_makeGetter
    

    const bool markovChainEnded() const
    {
        return chainStat.chainEndingReason != NULL;
    }
    
    struct MarkovChainStats;
    const MarkovChainStats& markovChainStats() const
    {
        return chainStat;
    }
    
    struct MarkovChainStats {
        
        MarkovChainStats()
        : stdDev(0), avgFitness(0), temperature(-1), currentFitness(-1), alpha(0), delta(0),
          //equilibriumNotReached(false), negativeTemp(false), terminated(false), markovChainEnded(false)
          equilibriumNotReached(false), negativeTemp(false), terminated(false)
        { }
        
        double
            stdDev,
            avgFitness,
            //prevAvgFitness,
            //expectedDecreaseInCost, // delta
            //costs_sum,
            temperature,
            currentFitness,
            alpha,
            delta
        ;
        int generatedSolutions, acceptedSolutions;
        //bool reinitializing, terminated, markovChainEnded;
        bool equilibriumNotReached, negativeTemp, terminated; //, markovChainEnded;
        const char * stoppingReason;
        const char * chainEndingReason; // if NULL, the chain has not ended
        //EOT& currentSolution;
        //moTrikiCoolingSchedule& coolingSchedule;
        void print(std::ostream& os = std::cout, bool onlyWhenChainEnds = true) const
        {
            //if (markovChainEnded || !onlyWhenChainEnds)
            if (chainEndingReason != NULL || !onlyWhenChainEnds)
            {
                //os << "T=" << prevTemp << "  avgFitness=" << prevAvgFitness << "  stdDev=" << stdDev << "  currentFitness=" << currentSolution.fitness() << endl;
                os << "T=" << temperature << "  avgFitness=" << avgFitness << "  stdDev=" << stdDev << "  currentFitness=" << currentFitness << "  expected decrease in cost=" << delta << endl;
                //os << "T=" << prevTemp << " \t\tavgFitness=" << prevAvgFitness << " \t\tstdDev=" << stdDev << " \t\tcurrentFitness=" << currentFitness << endl;
                //os << "T*=" << alpha << "  reinitializing=" << reinitializing << "  markovChainEnded=" << markovChainEnded << endl;// << "  terminated=" << terminated;
                //os << "T*=" << alpha << "  markovChainEnded=" << markovChainEnded;// << "  terminated=" << terminated;
                // TODONE print delta ? (expected decrease in cost)
                if (chainEndingReason != NULL)
                    os << "T*=" << alpha << "  chain ended, because " << chainEndingReason;
                if (equilibriumNotReached)
                    os << "  /!\\ equilibrium not reached";
                if (negativeTemp)
                    os << "  /!\\ negative temperature";
                os << endl;
                if (terminated)
                    os << "Terminated, because " << stoppingReason << endl;
                //    os << endl;
            }
        }
    };
    

private:
//public://FIXME add friend
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
    /*
    double
        stdDev,
        prevAvgFitness,
        expectedDecreaseInCost, // delta
        costs_sum,
        prevTemp
    ;*/
    MarkovChainStats chainStat;
    //double costs_sum;
    
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
    //bool reinitializing, terminated, markovChainEnded;
    
    //moFitnessVarianceStat<EOT> varStat;
    moFitnessMomentsStat<EOT> momentStat;
    bool statIsInitialized, reinitializing;
    
    //ofstream outf;
    /*
    static const char * stoppingReasons[] = {"no accepted solution", "null std dev" , "frozen >= theta"};
    static const char * chainEndingReasons[] = {"MAX GENerated solutions", "MAX ACCepted solutions"};
    */
    static const char * stoppingReasons[];
    static const char * chainEndingReasons[];
    
protected:
    
//    class Monitor {
//    public:
//        
//        Monitor(moTrikiCoolingSchedule& _cooling)
//        : cooling(_cooling)
//        { }
//        
//        /*void setTemperatureOstream(ostream& os)
//        {
//            
//        }*/
//        void getLatestTemperature()
//        {
//            return cooling.prevTemp;
//        }
//        
//        void printCurrentStatus(ostream& os)
//        {
//            if (accepted >= max_accepted || generated >= max_generated)
//            {
//                os << "Markov chain finished. Temp was " << getLatestTemperature(); // chain number
//                
//            }
//        }
//        
//    private:
//        moTrikiCoolingSchedule& cooling;
//    };
    
};



template< class Neighbor >
const char * moTrikiCoolingSchedule<Neighbor>::stoppingReasons[] = {"no accepted solution", "null std dev" , "frozen >= theta"};

template< class Neighbor >
//const char * moTrikiCoolingSchedule<Neighbor>::chainEndingReasons[] = {"MAX GENerated solutions", "MAX ACCepted solutions"};
const char * moTrikiCoolingSchedule<Neighbor>::chainEndingReasons[] = {"MAX ACCepted solutions", "MAX GENerated solutions"};




#endif










