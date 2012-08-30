/*

(c) 2010 Thales group

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; version 2
    of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>

*/

#ifndef _eoFeasibleRatioStat_h_
#define _eoFeasibleRatioStat_h_

#include <algorithm>

#include <eoDualFitness.h>
#include <utils/eoLogger.h>

#include "eoStat.h"

/** Ratio of the number of individuals with a feasible dual fitness in the population (@see eoDualFitness)
 *
 * @ingroup Stats
 */
template<class EOT>
class eoFeasibleRatioStat : public eoStat< EOT, double >
{
public:
    using eoStat<EOT, double>::value;

    eoFeasibleRatioStat( std::string description = "FeasibleRatio" ) : eoStat<EOT,double>( 0.0, description ) {}

    virtual void operator()( const eoPop<EOT> & pop )
    {
#ifndef NDEBUG
        assert( pop.size() > 0 );

        double count = static_cast<double>( std::count_if( pop.begin(), pop.end(), eoIsFeasible<EOT> ) );
        double size = static_cast<double>( pop.size() );
        double ratio = count/size;
        eo::log << eo::xdebug << "eoFeasibleRatioStat: " << count << " / " << size << " = " << ratio << std::endl;
        value() = ratio;
#else
        value() = static_cast<double>( std::count_if( pop.begin(), pop.end(), eoIsFeasible<EOT> ) ) / static_cast<double>( pop.size() );
#endif
    }

 virtual std::string className(void) const { return "eoFeasibleRatioStat"; }
};

#endif // _eoFeasibleRatioStat_h_
