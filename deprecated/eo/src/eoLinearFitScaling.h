/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoLinearFitScaling.h
   (c) GeNeura Team, 1998, Maarten Keijzer, Marc Schoenauer, 2001

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoLinearFitScaling_h
#define eoLinearFitScaling_h

#include <eoSelectFromWorth.h>
#include <eoPerf2Worth.h>

/** An instance of eoPerf2Worth
 *  COmputes the linearly scaled fitnesses
 *  with given selective pressure
 *  Pselect(Best) == pressure/sizePop
 *  Pselect(average) == 1.0/sizePop
 *  truncate negative values to 0 -
 *
 * to be used within an eoSelectFromWorth object
 *
 * @ingroup Selectors
 */
template <class EOT>
class eoLinearFitScaling : public eoPerf2Worth<EOT> // false: do not cache fitness
{
public:

    using eoPerf2Worth<EOT>::value;

    /* Ctor:
       @param _p selective pressure (in (1,2])
       @param _e exponent (1 == linear)
    */
    eoLinearFitScaling(double _p=2.0)
        : pressure(_p) {}

    /* COmputes the ranked fitness: fitnesses range in [m,M]
       with m=2-pressure/popSize and M=pressure/popSize.
       in between, the progression depends on exponent (linear if 1).
    */
    virtual void operator()(const eoPop<EOT>& _pop) {
        unsigned pSize =_pop.size();
        // value() refers to the vector of worthes (we're in an eoParamvalue)
        value().resize(pSize);

        // best and worse fitnesses
        double bestFitness = static_cast<double> (_pop.best_element().fitness());
        //    double worstFitness = static_cast<double> (_pop.worse_element().fitness());

        // average fitness
        double sum=0.0;
        unsigned i;
        for (i=0; i<pSize; i++)
            sum += static_cast<double>(_pop[i].fitness());
        double averageFitness = sum/pSize;

        // the coefficients for linear scaling
        double denom = pSize*(bestFitness - averageFitness);
        double alpha = (pressure-1)/denom;
        double beta = (bestFitness - pressure*averageFitness)/denom;

        for (i=0; i<pSize; i++) { // truncate to 0
            value()[i] = std::max(alpha*_pop[i].fitness()+beta, 0.0);
        }
    }

private:
    double pressure;	// selective pressure
};



#endif
