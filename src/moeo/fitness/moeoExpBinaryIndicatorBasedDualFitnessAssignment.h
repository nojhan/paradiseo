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

#include "moeoExpBinaryIndicatorBasedFitnessAssignment.h"

template<class MOEOT>
class moeoExpBinaryIndicatorBasedDualFitnessAssignment : public moeoExpBinaryIndicatorBasedFitnessAssignment<MOEOT>
{
protected:
    eoDualPopSplit<MOEOT> _pop_split;

public:
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    typedef typename ObjectiveVector::Type Type;
    typedef typename MOEOT::Fitness Fitness;

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

        // if there is at least one feasible individual,
        // it will supersede all the unfeasible ones
        if( _pop_split.unfeasible().size() != 0 ) {
            this->setup(_pop_split.unfeasible());
            this->computeValues(_pop_split.unfeasible());
            this->setFitnesses(_pop_split.unfeasible()); // NOTE: this alter individuals
        }

        if( _pop_split.feasible().size() != 0 ) {
            this->setup(_pop_split.feasible());
            this->computeValues(_pop_split.feasible());
            this->setFitnesses(_pop_split.feasible()); // NOTE: this alter individuals
        }

        // bring back altered individuals in the pop
        // pop = _pop_split.merge();

        eoPop<MOEOT> merged = _pop_split.merge();
        assert( pop.size() == merged.size());
        for( unsigned int i=0; i<pop.size(); ++i ) {
            pop[i] = merged[i];
        }
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
                } else { // if i == j
                    assert( i == j );
                    values[i][j] = Type( 0.0, pop[i].objectiveVector().is_feasible() );
                }
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

    virtual Fitness computeFitness(const unsigned int _idx)
    {
        // Fitness result( 0.0, values[_idx][_idx].is_feasible() );
        Fitness result( 0.0, values[_idx][_idx].is_feasible() );
        for (unsigned int i=0; i<values.size(); i++) { // i in pop.size()
            if (i != _idx) {
                result -= exp(-values[i][_idx]/kappa);
                // result += values[i][_idx];
            }
        }
        return confine(result);
    }

    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector objVec into account.
     * @param _pop the population
     * @param objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & pop, ObjectiveVector & objVec)
    {
        _pop_split(pop);

        if( objVec.is_feasible() ) {
            setup(_pop_split.feasible());
            updateFitnessByDeleting( _pop_split.feasible(), objVec);
        } else {
            setup(_pop_split.unfeasible());
            updateFitnessByDeleting( _pop_split.unfeasible(), objVec );
        }
        // pop = _pop_split.merge();
        eoPop<MOEOT> merged = _pop_split.merge();
        assert( pop.size() == merged.size());
        for( unsigned int i=0; i<pop.size(); ++i ) {
            pop[i] = merged[i];
        }

    }

protected:
    void updateFitnessByDeleting( eoPop < MOEOT > & pop, ObjectiveVector & objVec )
    {
        std::vector < double > v;
        v.resize(pop.size());
        for (unsigned int i=0; i<pop.size(); i++)
        {
            v[i] = metric(objVec, pop[i].objectiveVector());
        }
        for (unsigned int i=0; i<pop.size(); i++)
        {
            pop[i].fitness( confine( pop[i].fitness() + exp(-v[i]/kappa) ), pop[i].is_feasible() );
        }
    }

    template<class T>
    T confine( T n )
    {
        T tmax = std::numeric_limits<typename T::Base>::max();
        T tmin = -1 * tmax;

        tmin.is_feasible( n.is_feasible() );
        tmax.is_feasible( n.is_feasible() );

        if( n < tmin ) {
            return tmin;
        } else if( n > tmax ) {
            return tmax;
        } else {
            return n;
        }
    }

};

#endif // MOEOEXPBINARYINDICATORBASEDDUALFITNESSASSIGNMENT_H_
