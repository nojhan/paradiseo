// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRankingSelect.h
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

#ifndef eoRankingSelect_h
#define eoRankingSelect_h

//-----------------------------------------------------------------------------

#include <utils/eoRNG.h>
#include <utils/selectors.h>
#include <eoSelectOne.h>

//-----------------------------------------------------------------------------
/** eoRankingSelect: select an individual proportional to its rank
          this is actually the linearRanking
*/
//-----------------------------------------------------------------------------

template <class EOT> class eoRankingSelect: public eoSelectOne<EOT> 
{
public:
  /** Ctor:
   *  @param _p the selective pressure, should be in [1,2] (2 is the default)
   *  @param _pop an optional population
   */
  eoRankingSelect(double _p = 2.0, const eoPop<EOT>& _pop = eoPop<EOT>()): 
    pressure(_p), rank(0), rankFitness(0) 
  {
    if (_pop.size() > 0)
      {
	setup(_pop);
      }
  }

  // COmputes the coefficients of the linear transform uin such a way that
  // Pselect(Best) == Pselect(sizePop) == pressure/sizePop
  // Pselect(average) == 1.0/sizePop
  // Pselect(Worst == Pselect(1 == (2-pressure)/sizePop
  void setup(const eoPop<EOT>& _pop)
  {
    _pop.sort(rank);
    unsigned pSize =_pop.size();
    rankFitness.resize(pSize);
    double alpha = (2*pressure-2)/(pSize*(pSize-1));
    double beta = (2-pressure)/pSize;
    for (unsigned i=0; i<pSize; i++)
      {
	rankFitness[i] = alpha*(pSize-1-i)+beta;
      }
  }
    
  /** do the selection, call roulette_wheel on rankFitness
   */
  const EOT& operator()(const eoPop<EOT>& _pop) 
  {
    unsigned selected = rng.roulette_wheel(rankFitness);
    return *(rank[selected]);
  }

private :
  double pressure;
  vector<const EOT *> rank;
  vector<double> rankFitness;
};

#endif 

