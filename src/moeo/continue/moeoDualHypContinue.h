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

#ifndef _moeoDualHypContinue_h
#define _moeoDualHypContinue_h

#include "moeoHypContinue.h"

/**
  Continues until the (feasible or unfeasible) given Pareto set is reached.


  @ingroup Continuators
  */
template< class MOEOT, class MetricT = moeoDualHyperVolumeDifferenceMetric<typename MOEOT::ObjectiveVector> >
class moeoDualHypContinue: public moeoHypContinue<MOEOT, MetricT >
{
protected:
    bool is_feasible;

    using moeoHypContinue<MOEOT, MetricT>::arch;
    using moeoHypContinue<MOEOT, MetricT>::OptimSet;

    using moeoHypContinue<MOEOT, MetricT>::pareto;
    using moeoHypContinue<MOEOT, MetricT>::is_null_hypervolume;

public:
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    typedef typename ObjectiveVector::Type AtomType;

    /** A continuator that stops once a given Pareto front has been reached
     *
     * You should specify the feasibility of the targeted front.
     * NOTE: the MOEOT::ObjectiveVector is supposed to implement the moeoDualRealObjectiveVector interface.
     *
     */
    moeoDualHypContinue( const std::vector<AtomType> & _OptimVec, bool _is_feasible, moeoArchive < MOEOT > & _archive,  bool _normalize=true, double _rho=1.1 )
        : moeoHypContinue<MOEOT, MetricT>( _OptimVec, _archive, _normalize, _rho ),
        is_feasible(_is_feasible)
    {
        assert( _OptimVec.size() > 0);
        vectorToParetoSet(_OptimVec);
    }

    /** A continuator that stops once a given Pareto front has been reached
     *
     * You should specify the feasibility of the targeted front.
     * NOTE: the MOEOT::ObjectiveVector is supposed to implement the moeoDualRealObjectiveVector interface.
     *
     */
    moeoDualHypContinue( const std::vector<AtomType> & _OptimVec, bool _is_feasible, moeoArchive < MOEOT > & _archive,  bool _normalize=true, ObjectiveVector& _ref_point=NULL )
        : moeoHypContinue<MOEOT, MetricT>( _OptimVec, _archive, _normalize, _ref_point ),
        is_feasible(_is_feasible)
    {
        assert( _OptimVec.size() > 0);
        vectorToParetoSet(_OptimVec);
    }


    /** Returns false when a ParetoSet is reached. */
    virtual bool operator() ( const eoPop<MOEOT>& /*_pop*/ )
    {
        std::vector<ObjectiveVector> bestCurrentParetoSet = pareto( arch );

#ifndef NDEBUG
        assert( bestCurrentParetoSet.size() > 0 );
        for( unsigned int i=1; i<bestCurrentParetoSet.size(); ++i ) {
            assert( bestCurrentParetoSet[i].is_feasible() == bestCurrentParetoSet[0].is_feasible() );
        }
#endif

        // The current Pareto front is either feasible or unfeasible.
        // It could not contains both kind of objective vectors, because a feasible solution always dominates an unfeasible front.
        if( bestCurrentParetoSet[0].is_feasible() != OptimSet[0].is_feasible() ) {
            return false;
        }

        return is_null_hypervolume( bestCurrentParetoSet );
    }

protected:

    /** Translate a vector given as param to the ParetoSet that should be reached. */
    virtual void vectorToParetoSet(const std::vector<AtomType> & _OptimVec)
    {
        unsigned dim = (unsigned)(_OptimVec.size()/ObjectiveVector::Traits::nObjectives());
        OptimSet.resize(dim);

        unsigned k=0;
        for(size_t i=0; i < dim; i++) {
            for (size_t j=0; j < ObjectiveVector::Traits::nObjectives(); j++) {
                // Use the feasibility declaration of an eoDualFitness
                OptimSet[i][j] = AtomType(_OptimVec[k++], is_feasible);
            }
        }
    }
};

#endif
