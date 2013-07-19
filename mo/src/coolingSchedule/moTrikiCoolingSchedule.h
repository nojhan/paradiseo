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

#include <neighborhood/moNeighborhood.h>

#include <continuator/moNeighborhoodStat.h>
#include <continuator/moStdFitnessNeighborStat.h>
#include <continuator/moStat.h>
#include <continuator/moFitnessMomentsStat.h>

/**
 * Cooling Schedule, adapted from E.Triki, Y.Collette, P.Siarry (2004)
 * This CS is based on an initial estimation of the standard deviation
 * and an expected decrease in cost between each Markov chain,
 * possibly re-estimating the prameters
 * 
 * 
 * A detailed explanation follows:
 * 
 * 
 *    Initialization
 * 
 * A random walk of n steps should be performed to estimate the std dev
 * of the fitness. The init temp is set to this value.
 * 
 * 
 *    Algorithm
 * 
 * The CS is based on Markov chains, during which the temp is constant.
 * A Markov chain ends when a given number of solutions 'max_accepted'
 * has been reached, or when a given number 'max_generated' of solutions
 * have been generated.
 * 
 * After each chain, the average cost of the solutions of the chain is
 * expected to have decreased of delta (compared to the previous chain)
 * with delta initialized to = stddev/mu2.
 * If it's the case (ie: avgCost/(prevAvgCost-delta) < xi, where xi == 1+epsilon)
 * then we say we're at equilibrium and we apply a normal temperature
 * decrease, of ratio
 *         alpha = 1-_temp*delta/variance
 * 
 * Negative temperatures (when alpha < 0) happen when the initial std
 * dev was misestimated or, according to the article, when "the minimum
 * cost is greater than the current average cost minus delta"(sic)(??).
 * In case of a neg temp, we increment a counter 'negative_temp' and we
 * reinitialize the algorithm until the temperature is no more negative,
 * unless the counter of neg temp reaches a constant K2, in which case
 * the behavior has been chosen to be this of a greedy algorithm (SA
 * with nil temperature).
 * 
 * Reinitializing the algo consists in:
 *          - setting the temperature "decrease" factor alpha to a
 *          constant lambda1 > 1 in order to in fact increase it
 *          - setting delta to sigma/mu1 (sigma being the std dev of the
 *          current Markov chain)
 *  
 *  Note that when not reinitializing, the expected decrease in cost
 *  'delta' is never supposed to change.
 *  
 *  If the eq is not reached after the current chain, we increment a
 *  counter 'equilibrium_not_reached', and when it reaches K1 we
 *  reinitialize the algorithm and reset the counter to 0 (resetting
 *  to 0 was an added behavior and not part of the article; but without
 *  it the algo got trapped).
 *
 *
 *    Termination
 * 
 * Currently, the algo terminates when the current average cost stops
 * changing for 'theta' chains, or when the current std dev becomes
 * null (added behavior; indeed, when the std dev is null it is no more
 * possible to compute alpha), or when there is no accepted solution
 * for the current "chain" (added, cf in this case we can't compute a
 * std dev or an average).
 * In practice, the algorithm never seems to terminate by "freezing"
 * (first case), obviously because we need an implementation of
 * approximate double comparison instead of exact comparison.
 * 
 * 
 */
template< class EOT >
class moTrikiCoolingSchedule: public moCoolingSchedule< EOT >
{
public:
    
    /**
     * Constructor for the cooling schedule
     * @param _initTemp the temperature at which the CS begins; a recommended value is _stdDevEstimation
     * @param _stdDevEstimation an estimation of the standard deviation of the fitness. Typically, a random walk of n steps is performed to estimate the std dev of the fitness
     * @param _max_accepted maximum number of solutions to accept before ending the Markov chain and reducing the temperature; depends on the pb/neighborhood
     * @param _max_generated maximum number of solutions to generate before ending the Markov chain and reducing the temperature; depends on the pb/neighborhood
     * @param _mu2 target decrease in cost factor, mu2 typically belongs to [1; 20]
     * @param _mu1 target decrease in cost factor when reinitializing, in [2; 20]
     * @param _lambda1 the increase in temperature (reheating factor), typically in [1.5; 4]
     * @param _lambda2 lambda2 in [0.5; 0.99]
     * @param _xi typically belongs to [1; 1.1]
     * @param _theta typically set to 10
     * @param _K1 in [1; 4], the number of chains without reaching equilibrium before we raise the temperature
     * @param _K2 maximul number of consecutive negative temperatures before switching to a greedy algorithm
     */
    moTrikiCoolingSchedule (
        double _initTemp,
        double _stdDevEstimation,
        int    _max_accepted = 50,
        int    _max_generated = 100,
        double _mu2 = 2.5,
        double _mu1 = 10,
        double _lambda1 = 2,
        double _lambda2 = .7,
        double _xi = 1.05,
        int    _theta = 10,
        int    _K1 = 10,
        int    _K2 = 5
    )
    : initTemp(_initTemp),
      initStdDev(_stdDevEstimation),
      mu2(_mu2),
      K1(_K1),
      K2(_K2),
      lambda1(_lambda1),
      lambda2(_lambda2),
      mu1(_mu1),
      xi(_xi),
      max_accepted(_max_accepted),
      max_generated(_max_generated),
      theta(_theta),
      statIsInitialized(false)
    {
        chainStat.temperature = initTemp;
    }
    
    /**
     * Initialization
     * @param _solution initial solution
     */
    double init(EOT & _solution) {

        chainStat.temperature = initTemp;
        
        accepted = generated = 0;
        
        negative_temp = equilibrium_not_reached = frozen = 0;
        
        chainStat.delta = initStdDev/mu2;
        
        reinitializing = false;
        
        return initTemp;
    }

    /**
     * update the temperature by a factor
     * @param _temp current temperature to update
     * @param _acceptedMove true when the move is accepted, false otherwise
     */
    void update(double& _temp, bool _acceptedMove, EOT & _solution) {
        
        /*
         * In the following code, things were added or modified from
         * the original (incomplete) version of the algorithm
         * described in [2004, Triki et al.]
         * Each added/modified behavior is labelled
         * with a "// ADDED!" comment.
         */
        
        chainStat.temperature = _temp;
        chainStat.stoppingReason = NULL;
        chainStat.chainEndingReason = NULL;
        chainStat.equilibriumNotReached = false;
        chainStat.negativeTemp = false;
        chainStat.generatedSolutions = generated;
        chainStat.acceptedSolutions = accepted;
        
        generated++;
        
        if (_acceptedMove)
        {
            accepted++;
            if (statIsInitialized)
                 momentStat(_solution);
            else momentStat.init(_solution), statIsInitialized = true;
        }
        
        if (accepted > max_accepted || generated > max_generated) {
            
            chainStat.chainEndingReason = accepted > max_accepted ? chainEndingReasons[0]: chainEndingReasons[1];
            
            double avgFitness = momentStat.value().first;
            double prevAvgFitness = chainStat.avgFitness;
            
            double alpha = 0;
            
            if (accepted == 0) // ADDED! Otherwise the computed std dev is null; we're probably at equilibrium
            {
                chainStat.stoppingReason = stoppingReasons[0];
                
                // Note: we could also not stop and just become greedy (temperature set to 0)
            }
            else
            {
                double avgFitness = momentStat.value().first;
                double variance = momentStat.value().second;
                chainStat.stdDev = sqrt(variance);
                double sigma = chainStat.stdDev;
                
                accepted = generated = 0;
                statIsInitialized = false;
                
                if (negative_temp < K2)
                {
                    if (!reinitializing)
                    {
                        if (avgFitness/(prevAvgFitness-chainStat.delta) > xi)
                             equilibrium_not_reached++, chainStat.equilibriumNotReached = true;
                        else equilibrium_not_reached = 0;
                    }
                    if (equilibrium_not_reached > K1)
                    {
                        reinitializing = true;
                        
                        alpha = lambda1;
                        chainStat.delta = sigma/mu1;
                        equilibrium_not_reached = 0; // ADDED! Otherwise the algo gets trapped here!
                    }
                    else if (_temp*chainStat.delta/(sigma*sigma) >= 1)
                    {
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
                    else
                    {
                        reinitializing = false;
                        alpha = 1-_temp*chainStat.delta/variance;
                        
                        if (sigma == 0) // ADDED! When std dev is null, the solution is probably at eq, and the algo can't go on anyways
                            chainStat.stoppingReason = stoppingReasons[1];
                    }
                }
                else
                { /* Note: the paper doesn't specify a value for alpha in this case.
                     We've chosen to let it set to 0, which means the algorithm becomes greedy. */
                    alpha = 0;
                }
            }
            
            _temp *= alpha;
            
            chainStat.currentFitness = _solution.fitness();
            chainStat.alpha = alpha;
            chainStat.avgFitness = avgFitness;
            
            
            // TODO use a relative-epsilon comparison to approximate equality
            if (avgFitness == prevAvgFitness)
                 frozen++;
            else frozen = 0;
            
            if (frozen >= theta)
                chainStat.stoppingReason = stoppingReasons[2];
            
        }
        
    }
    
    /*
     * operator() Determines if the cooling schedule shall end or continue
     * @param temperature the current temperature
     */
    bool operator() (double temperature)
    {
        
        return frozen < theta
                && !chainStat.stoppingReason ; // ADDED! because 'frozen' isn't a sufficient terminating criterion (yet?)
        
    }
    
    /*
     * Definition of getter functions useful for monitoring the algorithm
     * using an eoGetterUpdater.
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
          equilibriumNotReached(false), negativeTemp(false)
        { }
        
        double
            stdDev,
            avgFitness,
            temperature,
            currentFitness,
            alpha,
            delta
        ;
        int generatedSolutions, acceptedSolutions;
        bool equilibriumNotReached, negativeTemp;
        const char * stoppingReason;    // if NULL, the algo has not stopped
        const char * chainEndingReason; // if NULL, the chain has not ended
        
        void print(std::ostream& os = std::cout, bool onlyWhenChainEnds = true) const
        {
            if (chainEndingReason != NULL || !onlyWhenChainEnds)
            {
                os << "T=" << temperature << "  avgFitness=" << avgFitness << "  stdDev=" << stdDev
                   << "  currentFitness=" << currentFitness << "  expected decrease in cost=" << delta
                   << std::endl;
                if (chainEndingReason)
                    os << "T*=" << alpha << "  chain ended, because " << chainEndingReason;
                if (equilibriumNotReached)
                    os << "  /!\\ equilibrium not reached";
                if (negativeTemp)
                    os << "  /!\\ negative temperature";
                os << std::endl;
                if (stoppingReason)
                    os << "Terminated, because " << stoppingReason << std::endl;
            }
        }
        
    };
    

private:
    
    // parameters of the algorithm
    
    const double
        initTemp,
        initStdDev,
        mu2,
        K1,
        K2,
        lambda1,
        lambda2,
        mu1,
        xi
    ;
    const int
        max_accepted,
        max_generated,
        theta
    ;
    
    // Variables of the algorithm
    
    MarkovChainStats chainStat;
    
    int
        accepted,
        generated,
        equilibrium_not_reached,
        negative_temp,
        frozen
    ;
    
    bool statIsInitialized, reinitializing;
    
    moFitnessMomentsStat<EOT> momentStat;
    
    
    // Possible reasons why the algorithm has stopped
    static const char * stoppingReasons[];
    
    // Possible reasons why the previous Markov chain has ended
    static const char * chainEndingReasons[];
    
};


/*
 * Definition of the static members of the class
 */
template< class Neighbor >
const char * moTrikiCoolingSchedule<Neighbor>::stoppingReasons[] = {"no accepted solution", "null std dev" , "frozen >= theta"};
template< class Neighbor >
const char * moTrikiCoolingSchedule<Neighbor>::chainEndingReasons[] = {"MAX ACCepted solutions", "MAX GENerated solutions"};


#endif



