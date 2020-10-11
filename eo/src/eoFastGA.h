
/*
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
*/

#ifndef _eoFastGA_H_
#define _eoFastGA_H_

/** The Fast Genetic Algorithm.
 *
 * @ingroup Algorithms
 */
template<class EOT>
class eoFastGA : public eoAlgo<EOT>
{
protected:
    double _rate_crossover;
    eoSelectOne<EOT>& _select_cross;
    eoQuadOp<EOT>& _crossover;
    eoSelectOne<EOT>& _select_aftercross;

    double _rate_mutation;
    eoSelectOne<EOT>& _select_mut;
    eoMonOp<EOT>& _mutation;

    eoPopEvalFunc<EOT>& _pop_eval;
    eoReplacement<EOT>& _replace;

    eoContinue<EOT>& _continuator;

    double _offsprings_size;

public:

    eoFastGA(
        double rate_crossover,
        eoSelectOne<EOT>& select_cross,
        eoQuadOp<EOT>& crossover,
        eoSelectOne<EOT>& select_aftercross,
        double rate_mutation,
        eoSelectOne<EOT>& select_mut,
        eoMonOp<EOT>& mutation,
        eoPopEvalFunc<EOT>& pop_eval,
        eoReplacement<EOT>& replace,
        eoContinue<EOT>& continuator,
        double offsprings_size = 0
    ) :
        _rate_crossover(rate_crossover),
        _select_cross(select_cross),
        _crossover(crossover),
        _select_aftercross(select_aftercross),
        _rate_mutation(rate_mutation),
        _select_mut(select_mut),
        _mutation(mutation),
        _pop_eval(pop_eval),
        _replace(replace),
        _continuator(continuator),
        _offsprings_size(offsprings_size)
    {
    }

    void operator()(eoPop<EOT>& pop)
    {
#ifndef NDEBUG
        assert(pop.size() > 0);
        for(auto sol : pop) {
            assert(not sol.invalid());
        }
#endif
        // Set lambda to the pop size
        // if it was not set up at construction.
        if(_offsprings_size == 0) {
            // eo::log << eo::debug << "Set offspring size to: " << pop.size() << std::endl;
            _offsprings_size = pop.size();
        }

        do {
            eoPop<EOT> offsprings;

            for(size_t i=0; i < _offsprings_size; ++i) {
                // eo::log << eo::xdebug << "\tOffspring #" << i << std::endl;

                if(eo::rng.flip(_rate_crossover)) {
                    // eo::log << eo::xdebug << "\t\tDo crossover" << std::endl;
                    // Manual setup of eoSelectOne
                    // (usually they are setup in a
                    // wrapping eoSelect).
                    _select_cross.setup(pop);

                    // Copy of const ref solutions,
                    // because one alter them hereafter.
                    EOT sol1 = _select_cross(pop);
                    EOT sol2 = _select_cross(pop);

                    // If the operator returns true,
                    // solutions have been altered.
                    if(_crossover(sol1, sol2)) {
                        sol1.invalidate();
                        sol2.invalidate();
                    }

                    // Select one of the two solutions
                    // which have been crossed.
                    eoPop<EOT> crossed; 
                    crossed.push_back(sol1);
                    crossed.push_back(sol2);
                    _select_aftercross.setup(crossed);
                    EOT sol3 = _select_aftercross(crossed);

                    // Additional mutation (X)OR the crossed/cloned solution.
                    if(eo::rng.flip(_rate_mutation)) {
                        // eo::log << eo::xdebug << "\t\tDo mutation" << std::endl;
                        if(_mutation(sol3)) {
                            sol3.invalidate();
                        }
                    }
                    offsprings.push_back(sol3);

                } else { // If not crossing, always mutate.
                    // eo::log << eo::xdebug << "\t\tNo crossover, do mutation" << std::endl;
                    _select_mut.setup(pop);
                    EOT sol3 = _select_mut(pop);
                    if(_mutation(sol3)) {
                        sol3.invalidate();
                    }
                    offsprings.push_back(sol3);
                }
            }
            assert(offsprings.size() == _offsprings_size);

            _pop_eval(pop, offsprings);
            _replace(pop, offsprings);

            // eo::log << eo::xdebug << "\tEnd of generation" << std::endl;

        } while(_continuator(pop));
#ifndef NDEBUG
        assert(pop.size() > 0);
        for(auto sol : pop) {
            assert(not sol.invalid());
        }
#endif
    }

};

#endif // _eoFastGA_H_
