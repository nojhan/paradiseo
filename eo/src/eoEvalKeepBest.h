
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

   © 2012 Thales group

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef eoEvalKeepBest_H
#define eoEvalKeepBest_H

#include <eoEvalFunc.h>
#include <utils/eoParam.h>

/**
  Evaluate with the given evaluator and keep the best individual found so far.

  This is useful if you use a non-monotonic algorithm, such as CMA-ES, where the
  population's best fitness can decrease between two generations. This is
  sometime necessary and one can't use elitist replacors, as one do not want to
  introduce a bias in the population.

  The eoEvalBestKeep is a wrapper around a classical evaluator, that keep the
  best individual it has found since its instanciation.

  To get the best individual, you have to call best_element() on the
  eoEvalKeepBest itself, and not on the population (or else you would get the
  best individual found at the last generation).

  Example:

    MyEval true_eval;
    eoEvalKeepBest<T> wrapped_eval( true_eval );

    // as an interesting side effect, you will get the best individual since
    // initalization.
    eoPop<T> pop( my_init );
    eoPopLoopEval<T> loop_eval( wrapped_eval );

    loop_eval( pop );

    eoEasyEA algo( …, wrapped_eval, … );

    algo(pop);

    // do not use pop.best_element()!
    std::cout << wrapped_eval.best_element() << std::endl;

  @ingroup Evaluation
  */
template<class EOT> class eoEvalKeepBest : public eoEvalFunc<EOT>, public eoValueParam<EOT>
{
    public :
        eoEvalKeepBest(eoEvalFunc<EOT>& _func, std::string _name = "VeryBest. ")
            : eoValueParam<EOT>(EOT(), _name), func(_func) {}

        virtual void operator()(EOT& sol)
        {
            if( sol.invalid() ) {
                func(sol); // evaluate

                // if there is no best kept
                if( this->value().invalid() ) {
                    // take the first individual as best
                    this->value() = sol;
                } else {
                    // if sol is better than the kept individual
                    if( sol.fitness() > this->value().fitness() ) {
                        this->value() = sol;
                    }
                }
            } // if invalid
        }

        //! Return the best individual found so far.
        EOT best_element()
        {
            return this->value();
        }

        /** Reset the best individual to the given one. If no individual is
         * provided, the next evaluated one will be taken as a reference.
         */
        void reset( const EOT& new_best = EOT() )
        {
            this->value() = new_best;
        }

    protected :
        eoEvalFunc<EOT>& func;
};

#endif
