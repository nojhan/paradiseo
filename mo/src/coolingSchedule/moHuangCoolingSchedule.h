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

#include <continuator/moNeighborhoodStat.h>
#include <continuator/moStdFitnessNeighborStat.h>
#include <neighborhood/moNeighborhood.h>
#include <continuator/moStat.h>
#include <continuator/moFitnessMomentsStat.h>


#include <iostream>
using namespace std;

//!
/*!
 */
template< class EOT >
class moHuangCoolingSchedule: public moCoolingSchedule< EOT >
{
public:
    //typedef typename Neighbor::EOT EOT ;
    //typedef moNeighborhood<Neighbor> Neighborhood ;

    //! Constructor
    /*!
     */
    
    moHuangCoolingSchedule (double _initTemp, double _stdDevEstimation, double _lambda = .7, double _finalTemp = .01)
    : initTemp(_initTemp),
      stdDevEstimation(_stdDevEstimation),
      lambda(_lambda),
      finalTemp(_finalTemp)
    { }

    /**
     * Initial temperature
     * @param _solution initial solution
     */
    double init(EOT & _solution) {
        
        return initTemp;
    }

    /**
     * update the temperature by a factor
     * @param _temp current temperature to update
     * @param _acceptedMove true when the move is accepted, false otherwise
     */
    void update(double& _temp, bool _acceptedMove, EOT & _solution) {
        
        _temp *= exp ( -lambda*_temp / stdDevEstimation ) ;
        
    }

    //! Function which proceeds to the cooling
    /*!
     */
    bool operator() (double temperature)
    {
        return temperature > finalTemp;
        
    }
    
    
private:
    
    const double
    // parameters of the algorithm
        initTemp,
        stdDevEstimation,
        lambda,
        finalTemp
    ;
    
};

#endif










