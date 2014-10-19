// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSwapMutation.h
// (c) GeNeura Team, 2000 - EEAAX 2000 - Maarten Keijzer 2000
// (c) INRIA Futurs - Dolphin Team - Thomas Legrand 2007
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
                 thomas.legrand@lifl.fr
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoSwapMutation_h
#define eoSwapMutation_h

//-----------------------------------------------------------------------------


/**
 * Swap two components of a chromosome.
 *
 * @ingroup Variators
 */
template<class Chrom> class eoSwapMutation: public eoMonOp<Chrom>
{
 public:

  /// CTor
  eoSwapMutation(const unsigned _howManySwaps=1): howManySwaps(_howManySwaps)
  {
        // consistency check
        if(howManySwaps < 1)
                throw std::runtime_error("Invalid number of swaps in eoSwapMutation");
  }

  /// The class name.
  virtual std::string className() const { return "eoSwapMutation"; }

  /**
   * Swap two components of the given chromosome.
   * @param chrom The cromosome which is going to be changed.
   */
  bool operator()(Chrom& chrom)
    {
      unsigned i, j;

      for(unsigned int swap = 0; swap < howManySwaps; swap++)
      {
            // generate two different indices
        i=eo::rng.random(chrom.size());
        do j = eo::rng.random(chrom.size()); while (i == j);

            // swap
            std::swap(chrom[i],chrom[j]);
      }
      return true;
    }

   private:
        unsigned int howManySwaps;
};
/** @example t-eoSwapMutation.cpp
 */

//-----------------------------------------------------------------------------
#endif
