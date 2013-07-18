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

#ifndef _moHuangCoolingSchedule_h
#define _moHuangCoolingSchedule_h

#include <coolingSchedule/moCoolingSchedule.h>

#include <neighborhood/moNeighborhood.h>

#include <continuator/moNeighborhoodStat.h>
#include <continuator/moStdFitnessNeighborStat.h>
#include <continuator/moStat.h>
#include <continuator/moFitnessMomentsStat.h>

/**
 * Cooling Schedule, adapted from E.Triki, Y.Collette, P.Siarry (2004)
 * Fairly simple CS that is supposed to work fairly well.
 */
template< class EOT >
class moHuangCoolingSchedule: public moCoolingSchedule< EOT >
{
public:
    
    /**
     * Constructor for the cooling schedule
     * @param _initTemp the temperature at which the CS begins
     * @param _spanSize the number of steps to perform between each decrease of temperature
     * @param _lambda determines the decrease factor of the temperature, ie: alpha = e^(-lambda*T/stddev)
     * @param _finalTempDecrease 
     */
    moHuangCoolingSchedule (double _initTemp, int _spanSize, double _lambda = .7, double _finalTempDecrease = .995)
    : initTemp(_initTemp)
    , spanSize(_spanSize)
    , lambda(_lambda)
    , finalStdDev(_finalTempDecrease)
    { }
    
    /**
     * Initialization
     * @param _solution initial solution
     */
    double init(EOT & _solution) {
        statIsInitialized = terminated = false;
        step = 0;
        return initTemp;
    }
    
    /**
     * update the temperature by a factor
     * @param _temp current temperature to update
     * @param _acceptedMove true when the move is accepted, false otherwise
     */
    void update(double& _temp, bool _acceptedMove, EOT & _solution) {
        
        if (_acceptedMove)
        {
            if (statIsInitialized)
                 momentStat(_solution);
            else momentStat.init(_solution), statIsInitialized = true;
            
        }
        
        if (step >= spanSize)
        {
            step = 0;
            double alpha = exp( -lambda*_temp / sqrt(momentStat.value().second) );
            _temp *= alpha;
            terminated = alpha > finalStdDev;
        }
        
        step++;
    }
    
    /*
     * operator() Determines if the cooling schedule shall end or continue
     * @param temperature the current temperature
     */
    bool operator() (double temperature)
    {
        return !terminated;
    }
    
    
private:
    
    // parameters of the algorithm
    
    const double
       initTemp
     , lambda
     , finalStdDev
    ;
    const int spanSize;
    int step;
    
    
    // variables of the algorithm
    
    bool statIsInitialized, terminated;
    
    moFitnessMomentsStat<EOT> momentStat;
    
};

#endif










