/*

(c) 2013 Thales group

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

#ifndef _MOEOBINARYMETRICSTAT_H_
#define _MOEOBINARYMETRICSTAT_H_

#include <eo>

/** A wrapper to save a moeoMetric in an eoStat
 *
 * This wrap a MOEO binary metric into an eoStat
 * This is useful if you want to use it in a checkpoint, for instance.
 */
template <class MOEOT, class T = double>
class moeoBinaryMetricStat : public eoStat<MOEOT, T>
{
public:
    /** The objective vector type of a solution */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    moeoBinaryMetricStat(
            moeoVectorVsVectorBinaryMetric<ObjectiveVector, T> & metric,
            std::string description,
            T default_value = 0
        ) :
            eoStat<MOEOT,T>( default_value, description),
            _metric(metric),
            _first_gen(true)
    {}

    virtual std::string className(void) const
        { return "moeoBinaryMetricStat"; }


    virtual void operator()( const eoPop<MOEOT> & pop )
    {
        if( pop.size() ) {
            if( _first_gen ) {
                _first_gen = false;
            } else {
              // creation of the two Pareto sets
              std::vector < ObjectiveVector > from;
              std::vector < ObjectiveVector > to;
              for (unsigned int i=0; i<pop.size(); i++) {
                from.push_back( pop[i].objectiveVector() );
              }
              for (unsigned int i=0 ; i<_prev_pop.size(); i++) {
                to.push_back( _prev_pop[i].objectiveVector() );
              }

              // compute and save
              this->value() = _metric(from,to);
            } // if first gen

            // copy the pop
            _prev_pop = pop;
        } // if pop size
    }

protected:
    /** binary metric comparing two Pareto sets */
    moeoVectorVsVectorBinaryMetric<ObjectiveVector, T> & _metric;

    /** (n-1) population */
    eoPop<MOEOT> _prev_pop;

    /** is it the first generation ? */
    bool _first_gen;

};

#endif // _MOEOBINARYMETRICSTAT_H_
