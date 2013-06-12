/*
 * <moeoHypContinue.h>
 * Copyright (C) TAO Project-Team, INRIA-Saclay, 2011-2012
 * (C) TAO Team, LRI, 2011-2012
 *
 Mostepha-Redouane Khouadjia <mostepha-redouane.khouadjia@inria.fr>

 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 * ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 * Contact: paradiseo-help@lists.gforge.inria.fr
 *
 */
//-----------------------------------------------------------------------------



#ifndef _moeoHypContinue_h
#define _moeoHypContinue_h

#include <eoContinue.h>
#include <utils/eoLogger.h>
#include <metric/moeoHyperVolumeMetric.h>
#include <archive/moeoUnboundedArchive.h>

/**
  Continues until the given ParetoSet level is reached.

  @ingroup Continuators
  */
template< class MOEOT, class MetricT = moeoHyperVolumeDifferenceMetric<typename MOEOT::ObjectiveVector> >
class moeoHypContinue: public eoContinue<MOEOT>
{
public:

    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    typedef typename ObjectiveVector::Type AtomType;

    /// Ctor
    moeoHypContinue(  const std::vector<AtomType> & _OptimVec, moeoArchive < MOEOT > & _archive,  bool _normalize=true, double _rho=1.1)
        : eoContinue<MOEOT>(), arch(_archive), metric(_normalize,_rho)
    {
        vectorToParetoSet(_OptimVec);
    }

    moeoHypContinue( const std::vector<AtomType> & _OptimVec, moeoArchive < MOEOT > & _archive,  bool _normalize=true, ObjectiveVector& _ref_point=NULL)
        : eoContinue<MOEOT> (), arch(_archive), metric(_normalize,_ref_point)
    {
        vectorToParetoSet(_OptimVec);
    }


    /** Returns false when a ParetoSet is reached. */
    virtual bool operator() ( const eoPop<MOEOT>& /*_pop*/ )
    {
        std::vector<ObjectiveVector> bestCurrentParetoSet = pareto( arch );

        return is_null_hypervolume( bestCurrentParetoSet );
    }

    virtual std::string className(void) const { return "moeoHypContinue"; }

protected:

    std::vector<ObjectiveVector> pareto( moeoArchive<MOEOT> & _archive )
    {
        std::vector < ObjectiveVector > bestCurrentParetoSet;

        for (size_t i=0; i<arch.size(); i++) {
            bestCurrentParetoSet.push_back(arch[i].objectiveVector());
        }

        return bestCurrentParetoSet;
    }

    bool is_null_hypervolume( std::vector<ObjectiveVector>& bestCurrentParetoSet )
    {
        double hypervolume= metric( bestCurrentParetoSet, OptimSet );

        if (hypervolume==0) {
            eo::log << eo::logging << "STOP in moeoHypContinue: Best ParetoSet has been reached "
                << hypervolume << std::endl;
            return false;
        }
        return true;
    }

    /** Translate a vector given as param to the ParetoSet that should be reached. */
    virtual void vectorToParetoSet(const std::vector<AtomType> & _OptimVec)
    {
        unsigned dim = (unsigned)(_OptimVec.size()/ObjectiveVector::Traits::nObjectives());
        OptimSet.resize(dim);

        unsigned k=0;
        for(size_t i=0; i < dim; i++) {
            for (size_t j=0; j < ObjectiveVector::Traits::nObjectives(); j++) {
                OptimSet[i][j]=_OptimVec[k++];
            }
        }
    }

protected:
    moeoArchive <MOEOT> & arch;
    MetricT metric;
    std::vector <ObjectiveVector> OptimSet;
};


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
    moeoDualHypContinue<MOEOT, MetricT>( const std::vector<AtomType> & _OptimVec, bool _is_feasible, moeoArchive < MOEOT > & _archive,  bool _normalize=true, double _rho=1.1 )
        : moeoHypContinue<MOEOT, MetricT>( _OptimVec, _archive, _normalize, _rho ), is_feasible(_is_feasible)
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
    moeoDualHypContinue<MOEOT, MetricT>( const std::vector<AtomType> & _OptimVec, bool _is_feasible, moeoArchive < MOEOT > & _archive,  bool _normalize=true, ObjectiveVector& _ref_point=NULL )
        : moeoHypContinue<MOEOT, MetricT>( _OptimVec, _archive, _normalize, _ref_point ), is_feasible(_is_feasible)
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
