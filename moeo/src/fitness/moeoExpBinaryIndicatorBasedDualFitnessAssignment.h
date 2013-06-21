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


#ifndef MOEOEXPBINARYINDICATORBASEDDUALFITNESSASSIGNMENT_H_
#define MOEOEXPBINARYINDICATORBASEDDUALFITNESSASSIGNMENT_H_

#include <fitness/moeoExpBinaryIndicatorBasedFitnessAssignment.h>

template<class MOEOT>
class moeoExpBinaryIndicatorBasedDualFitnessAssignment : public moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>
{
protected:
    eoDualPopSplit<MOEOT> _pop_split;

public:
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    typedef typename ObjectiveVector::Type Type;

    using moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>::values;

    moeoExpBinaryIndicatorBasedDualFitnessAssignment(
            moeoNormalizedSolutionVsSolutionBinaryMetric<ObjectiveVector,double> & metric,
            const double kappa = 0.05
        ) : moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>( metric, kappa ) {}


    /*! If the population is homogeneous (only composed of feasible individuals or unfeasible ones),
     * then apply the operators on the whole population.
     * But, if there is at least one feasible individual, then apply them only on the feasible individuals.
     */
    virtual void operator()( eoPop<MOEOT>& pop )
    {
        // separate the pop in feasible/unfeasible
        _pop_split( pop );

        eoPop<MOEOT>* ppop;
        // if there is at least one feasible individual,
        // it will supersede all the unfeasible ones
        if( _pop_split.feasible().size() == 0 ) {
            ppop = & _pop_split.unfeasible();
        } else {
            ppop = & _pop_split.feasible();
        }

        this->setup(*ppop);
        this->computeValues(*ppop);
        this->setFitnesses(*ppop); // NOTE: this alter individuals

        // bring back altered individuals in the pop
        pop = _pop_split.merge();
    }


protected:

    using moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>::kappa;
    using moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>::metric;


    /**
     * Sets the bounds for every objective using the min and the max value for every objective vector of _pop
     * @param _pop the population
     */
    void setup(const eoPop < MOEOT > & _pop)
    {
        Type worst, best;
        typename MOEOT::ObjectiveVector::Type::Compare cmp;

        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            worst = _pop[0].objectiveVector()[i];
            best = _pop[0].objectiveVector()[i];
            for (unsigned int j=1; j<_pop.size(); j++)
            {
                // use the overloaded comparison operators
                worst = std::min(worst, _pop[j].objectiveVector()[i], cmp );
                best = std::max(best, _pop[j].objectiveVector()[i], cmp );
            }
            // Get real min/max
            double min = std::min(worst.value(), best.value());
            double max = std::max(worst.value(), best.value());

            // Build a fitness with them
            assert( best.is_feasible() == worst.is_feasible() ); // we are supposed to work on homogeneous pop
            Type fmin( min, best.is_feasible() );
            Type fmax( max, best.is_feasible() );

            // setting of the bounds for the objective i
            metric.setup( fmin, fmax, i);
        }
    }

    /**
     * Compute every indicator value in values (values[i] = I(_v[i], _o))
     * @param _pop the population
     */
    virtual void computeValues(const eoPop < MOEOT > & pop)
    {
        values.clear();
        values.resize(pop.size());
        for (unsigned int i=0; i<pop.size(); i++) {
            values[i].resize(pop.size());
            // the metric may not be symetric, thus neither is the matrix
            for (unsigned int j=0; j<pop.size(); j++) {
                if (i != j) {
                    values[i][j] = Type(
                            metric( pop[i].objectiveVector(), pop[j].objectiveVector() ),
                            pop[i].objectiveVector().is_feasible()
                        );
                } // if i != j
            } // for j in pop
        } // for i in pop
    }

    virtual void setFitnesses(eoPop < MOEOT > & pop)
    {
        for (unsigned int i=0; i<pop.size(); i++) {
            // We should maintain the feasibility of the fitness across computations
            pop[i].fitness( this->computeFitness(i), pop[i].fitness().is_feasible() );
        }
    }

    virtual Type computeFitness(const unsigned int _idx)
    {
      Type result( 0.0, values[_idx][_idx].is_feasible() );
      for (unsigned int i=0; i<values.size(); i++)
        {
          if (i != _idx)
            {
              result -= exp(-values[i][_idx]/kappa);
            }
        }
      return result;
    }


};

#endif // MOEOEXPBINARYINDICATORBASEDDUALFITNESSASSIGNMENT_H_
