// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFitnessScalingSelect.h
// (c) GeNeura Team, 1998, Maarten Keijzer 2000, Marc Schoenauer 2001
/* 
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoFitnessScalingSelect_h
#define eoFitnessScalingSelect_h

//-----------------------------------------------------------------------------

#include <utils/eoRNG.h>
#include <utils/selectors.h>
#include <eoSelectOne.h>

//-----------------------------------------------------------------------------
/** eoFitnessScalingSelect: select an individual proportional to its rank
          this is actually the linearRanking
*/
//-----------------------------------------------------------------------------

template <class EOT> class eoFitnessScalingSelect: public eoSelectOne<EOT> 
{
public:
     typedef typename EOT::Fitness Fitness;
  /** Ctor:
   *  @param _p the selective pressure, should be in [1,2] (2 is the default)
   *  @param _pop an optional population
   */
  eoFitnessScalingSelect(double _p = 2.0, const eoPop<EOT>& _pop = eoPop<EOT>()): 
    pressure(_p), scaledFitness(0) 
  {
    if (minimizing_fitness<EOT>())
      throw logic_error("eoFitnessScalingSelect: minimizing fitness");
    if (_pop.size() > 0)	   // a population in Ctor? initialize scaledFitness
      {
	setup(_pop);
      }
  }

  // COmputes the coefficients of the linear transform in such a way that
  // Pselect(Best) == pressure/sizePop
  // Pselect(average) == 1.0/sizePop
  // we truncate negative values to 0 - 
  // we could also adjust the pressure so that worst get 0 scaled fitness
  void setup(const eoPop<EOT>& _pop)
  {
    unsigned pSize =_pop.size();
    scaledFitness.resize(pSize);

    // best and worse fitnesses
    double bestFitness = static_cast<double> (_pop.best_element().fitness());
    double worstFitness = static_cast<double> (_pop.worse_element().fitness());

    // average fitness
    double sum=0.0;
    for (unsigned i=0; i<pSize; i++)
      sum += static_cast<double>(_pop[i].fitness());
    double averageFitness = sum/pSize;

    // the coefficients for linear scaling
    double denom = pSize*(bestFitness - averageFitness);
    double alpha = (pressure-1)/denom;
    double beta = (bestFitness - pressure*averageFitness)/denom;
//     if (beta < 0)
//       beta = max(beta, -alpha*worstFitness);
    for (unsigned i=0; i<pSize; i++) // truncate to 0
      {
	scaledFitness[i] = max(alpha*_pop[i].fitness()+beta, 0.0);
      }
  }
    
  /** do the selection, call roulette_wheel on rankFitness
   */
  const EOT& operator()(const eoPop<EOT>& _pop) 
  {
    unsigned selected = rng.roulette_wheel(scaledFitness);
    return _pop[selected];
  }

private :
  double pressure;
  vector<double> scaledFitness;
};

#endif 

