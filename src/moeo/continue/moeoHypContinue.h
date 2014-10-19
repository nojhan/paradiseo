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

#include "../../eo/eoContinue.h"
#include "../../eo/utils/eoLogger.h"
#include "../metric/moeoHyperVolumeMetric.h"
#include "../archive/moeoUnboundedArchive.h"

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
        : eoContinue<MOEOT>(), arch(_archive), default_metric(new MetricT(_normalize,_rho)), metric(*default_metric)
    {
        vectorToParetoSet(_OptimVec);
    }

    moeoHypContinue( const std::vector<AtomType> & _OptimVec, moeoArchive < MOEOT > & _archive,  bool _normalize=true, ObjectiveVector& _ref_point=NULL)
        : eoContinue<MOEOT>(), arch(_archive), default_metric(new MetricT(_normalize,_ref_point)), metric(*default_metric)
    {
        vectorToParetoSet(_OptimVec);
    }

    moeoHypContinue( MetricT& _metric, const std::vector<AtomType> & _OptimVec, moeoArchive < MOEOT > & _archive )
        : eoContinue<MOEOT>(), arch(_archive), default_metric(NULL), metric(_metric)
    {
        vectorToParetoSet(_OptimVec);
    }

    ~moeoHypContinue()
    {
        if( default_metric != NULL ) {
            delete default_metric;
        }
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

        for (size_t i=0; i<_archive.size(); i++) {
            bestCurrentParetoSet.push_back(_archive[i].objectiveVector());
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
    MetricT* default_metric;
    MetricT& metric;
    std::vector <ObjectiveVector> OptimSet;
};


#endif
