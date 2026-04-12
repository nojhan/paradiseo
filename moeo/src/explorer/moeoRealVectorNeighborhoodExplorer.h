/*
* <moeoRealVectorNeighborhoodExplorer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Eremey Valetov
*
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

#ifndef MOEOREALVECTORNEIGHBORHOODEXPLORER_H
#define MOEOREALVECTORNEIGHBORHOODEXPLORER_H

#include <eoPop.h>
#include <utils/eoRealVectorBounds.h>
#include <explorer/moeoPopNeighborhoodExplorer.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <random>
#include <cmath>

/**
 * Neighborhood explorer for real-valued vector solutions.
 * Generates random directions in the search space, evaluates neighbors,
 * and performs geometric scaling along improving directions.
 *
 * @tparam Neighbor neighbor type (must provide EOT typedef)
 */
template <class Neighbor>
class RealVectorNeighborhoodExplorer : public moeoPopNeighborhoodExplorer<Neighbor> {
public:
    typedef typename Neighbor::EOT MOEOT;

    /**
     * @param _eval evaluation function
     * @param _bounds bounds for each dimension
     * @param _epsilon step size relative to range (scalar, applied to all dimensions)
     * @param _ntrydirections number of random directions to try
     * @param _breakdirectionlooponfirstsuccess stop direction loop on first improving direction
     * @param _minntryscaling minimum scaling iterations before stopping
     * @param _maxntryscaling maximum scaling iterations
     * @param _scalingfactor multiplicative factor for scaling steps
     */
    RealVectorNeighborhoodExplorer(eoEvalFunc<MOEOT>& _eval, const eoRealVectorBounds& _bounds,
                                    double _epsilon = 1e-5, unsigned int _ntrydirections = 10,
                                    bool _breakdirectionlooponfirstsuccess = false,
                                    unsigned int _minntryscaling = 2, unsigned int _maxntryscaling = 5,
                                    unsigned int _scalingfactor = 4)
        : bounds(_bounds), epsilons(std::vector<double>(_bounds.size(), _epsilon)),
          ntrydirections(_ntrydirections), breakdirectionlooponfirstsuccess(_breakdirectionlooponfirstsuccess),
          minntryscaling(_minntryscaling), maxntryscaling(_maxntryscaling), scalingfactor(_scalingfactor),
          evalFunc(_eval), minntryscaling_vectype(false) {}

    /**
     * @param _eval evaluation function
     * @param _bounds bounds for each dimension
     * @param _epsilon step size relative to range (scalar)
     * @param _ntrydirections number of random directions
     * @param _breakdirectionlooponfirstsuccess stop on first success
     * @param _mintryscaling_explored per-visit-count minimum scaling vector
     * @param _maxntryscaling maximum scaling iterations
     * @param _scalingfactor multiplicative factor
     */
    RealVectorNeighborhoodExplorer(eoEvalFunc<MOEOT>& _eval, const eoRealVectorBounds& _bounds,
                                    double _epsilon, unsigned int _ntrydirections,
                                    bool _breakdirectionlooponfirstsuccess,
                                    std::vector<unsigned int> _mintryscaling_explored,
                                    unsigned int _maxntryscaling = 5, unsigned int _scalingfactor = 4)
        : bounds(_bounds), epsilons(std::vector<double>(_bounds.size(), _epsilon)),
          ntrydirections(_ntrydirections), breakdirectionlooponfirstsuccess(_breakdirectionlooponfirstsuccess),
          minntryscaling_explored(_mintryscaling_explored), maxntryscaling(_maxntryscaling),
          scalingfactor(_scalingfactor), evalFunc(_eval), minntryscaling_vectype(true) {}

    /**
     * @param _eval evaluation function
     * @param _bounds bounds for each dimension
     * @param _epsilons per-dimension step sizes
     * @param _ntrydirections number of random directions
     * @param _breakdirectionlooponfirstsuccess stop on first success
     * @param _minntryscaling minimum scaling iterations
     * @param _maxntryscaling maximum scaling iterations
     * @param _scalingfactor multiplicative factor
     */
    RealVectorNeighborhoodExplorer(eoEvalFunc<MOEOT>& _eval, const eoRealVectorBounds& _bounds,
                                    std::vector<double> _epsilons, unsigned int _ntrydirections = 10,
                                    bool _breakdirectionlooponfirstsuccess = false,
                                    unsigned int _minntryscaling = 2, unsigned int _maxntryscaling = 5,
                                    unsigned int _scalingfactor = 4)
        : bounds(_bounds), epsilons(_epsilons), ntrydirections(_ntrydirections),
          breakdirectionlooponfirstsuccess(_breakdirectionlooponfirstsuccess),
          minntryscaling(_minntryscaling), maxntryscaling(_maxntryscaling), scalingfactor(_scalingfactor),
          evalFunc(_eval), minntryscaling_vectype(false) {
        if (epsilons.size() != bounds.size())
            throw std::invalid_argument("The number of epsilons must match the number of bounds.");
    }

    /**
     * @param _eval evaluation function
     * @param _bounds bounds for each dimension
     * @param _epsilons per-dimension step sizes
     * @param _ntrydirections number of random directions
     * @param _breakdirectionlooponfirstsuccess stop on first success
     * @param _mintryscaling_explored per-visit-count minimum scaling vector
     * @param _maxntryscaling maximum scaling iterations
     * @param _scalingfactor multiplicative factor
     */
    RealVectorNeighborhoodExplorer(eoEvalFunc<MOEOT>& _eval, const eoRealVectorBounds& _bounds,
                                    std::vector<double> _epsilons, unsigned int _ntrydirections,
                                    bool _breakdirectionlooponfirstsuccess,
                                    std::vector<unsigned int> _mintryscaling_explored,
                                    unsigned int _maxntryscaling = 5, unsigned int _scalingfactor = 4)
        : bounds(_bounds), epsilons(_epsilons), ntrydirections(_ntrydirections),
          breakdirectionlooponfirstsuccess(_breakdirectionlooponfirstsuccess),
          minntryscaling_explored(_mintryscaling_explored), maxntryscaling(_maxntryscaling),
          scalingfactor(_scalingfactor), evalFunc(_eval), minntryscaling_vectype(true) {
        if (epsilons.size() != bounds.size())
            throw std::invalid_argument("The number of epsilons must match the number of bounds.");
    }

    void operator()(eoPop<MOEOT>& _src, std::vector<unsigned int> _select, eoPop<MOEOT>& _dest) override {
        for (unsigned int i = 0; i < _select.size(); i++) {
            explore(_src[_select[i]], _dest);
            _src[_select[i]].flag(_src[_select[i]].flag() + 1);
        }
        for (MOEOT& individual : _dest)
            individual.flag(0);
    }

protected:
    eoRealVectorBounds bounds;
    std::vector<double> epsilons;
    unsigned int ntrydirections;
    bool breakdirectionlooponfirstsuccess;
    bool minntryscaling_vectype;
    unsigned int minntryscaling;
    std::vector<unsigned int> minntryscaling_explored;
    unsigned int maxntryscaling;
    unsigned int scalingfactor;
    eoEvalFunc<MOEOT>& evalFunc;

    virtual void explore(MOEOT& _src, eoPop<MOEOT>& _dest) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(-1, 1);

        unsigned int minntryscaling_eff;
        if (minntryscaling_vectype) {
            if (!minntryscaling_explored.empty()) {
                minntryscaling_eff = minntryscaling_explored[
                    std::min(static_cast<size_t>(_src.flag()),
                             minntryscaling_explored.size() - 1)];
            } else {
                minntryscaling_eff = 1;
            }
        } else {
            minntryscaling_eff = minntryscaling;
        }

        for (unsigned int i = 0; i < ntrydirections; ++i) {
            MOEOT modified_individual = _src;
            std::vector<double> random_direction(modified_individual.size());

            double magnitude;
            do {
                magnitude = 0.0;
                for (unsigned int j = 0; j < _src.size(); ++j) {
                    random_direction[j] = distribution(generator);
                    magnitude += random_direction[j] * random_direction[j];
                }
            } while (magnitude > 1.0 || magnitude == 0);

            magnitude = std::sqrt(magnitude);

            for (unsigned int j = 0; j < modified_individual.size(); ++j) {
                double range = bounds.maximum(j) - bounds.minimum(j);
                random_direction[j] *= range * epsilons[j] / magnitude;
            }

            for (unsigned int j = 0; j < modified_individual.size(); ++j)
                modified_individual[j] += random_direction[j];
            modified_individual.invalidate();

            if (!bounds.isInBounds(modified_individual))
                continue;

            moeoParetoObjectiveVectorComparator<typename MOEOT::ObjectiveVector> comparator;
            evalFunc(modified_individual);

            if (comparator(_src.objectiveVector(), modified_individual.objectiveVector())) {
                _dest.push_back(modified_individual);

                MOEOT previous_explored_individual = modified_individual;
                bool continued_to_scale = false;

                for (unsigned int k = 0; k < maxntryscaling; ++k) {
                    for (unsigned int j = 0; j < modified_individual.size(); ++j)
                        random_direction[j] *= scalingfactor;

                    MOEOT scaled_individual = _src;
                    for (unsigned int j = 0; j < scaled_individual.size(); ++j)
                        scaled_individual[j] += random_direction[j];
                    scaled_individual.invalidate();

                    if (!bounds.isInBounds(scaled_individual))
                        break;

                    evalFunc(scaled_individual);

                    if (k == maxntryscaling - 1)
                        break;

                    if (comparator(_src.objectiveVector(), scaled_individual.objectiveVector())) {
                        _dest.push_back(scaled_individual);
                    } else if (k < minntryscaling_eff - 1) {
                        continued_to_scale = true;
                    } else {
                        break;
                    }

                    if (!comparator(scaled_individual.objectiveVector(), previous_explored_individual.objectiveVector())) {
                        if (k < minntryscaling_eff - 1)
                            continued_to_scale = true;
                        else
                            break;
                    }

                    previous_explored_individual = scaled_individual;
                }

                if (breakdirectionlooponfirstsuccess) break;
            }
        }
    }
};

#endif //MOEOREALVECTORNEIGHBORHOODEXPLORER_H
