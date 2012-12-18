// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoBitOp.h
// (c) GeNeura Team, 2000 - EEAAX 2000 - Maarten Keijzer 2000
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
 CVS Info: $Date: 2007-08-21 14:52:50 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/ga/eoBitOp.h,v 1.18 2007-08-21 14:52:50 kuepper Exp $ $Author: kuepper $
 */
//-----------------------------------------------------------------------------

#ifndef eoBitOp_h
#define eoBitOp_h

//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoRNG.h>
#include <eoInit.h>       // eoMonOp
#include <ga/eoBit.h>


/** eoOneBitFlip --> changes 1 bit
\class eoOneBitFlip eoBitOp.h ga/eoBitOp.h
\ingroup bitstring

@ingroup Variators
*/

template<class Chrom> class eoOneBitFlip: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  virtual std::string className() const { return "eoOneBitFlip"; }

  /**
   * Change one bit.
   * @param chrom The cromosome which one bit is going to be changed.
   */
  bool operator()(Chrom& chrom)
    {
      unsigned i = eo::rng.random(chrom.size());
      chrom[i] = !chrom[i];
      return true;
    }
};

/** eoDetBitFlip --> changes exactly k bits
\class eoDetBitFlip eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoDetBitFlip: public eoMonOp<Chrom>
{
 public:
  /**
   * (Default) Constructor.
   * @param _num_bit The number of bits to change
   * default is one - equivalent to eoOneBitFlip then
   */
  eoDetBitFlip(const unsigned& _num_bit = 1): num_bit(_num_bit) {}

  /// The class name.
  virtual std::string className() const { return "eoDetBitFlip"; }

  /**
   * Change num_bit bits.
   * @param chrom The cromosome which one bit is going to be changed.
   */
  bool operator()(Chrom& chrom)
    {
      // for duplicate checking see eoDetSingleBitFlip
      for (unsigned k=0; k<num_bit; k++)
        {
          unsigned i = eo::rng.random(chrom.size());
          chrom[i] = !chrom[i];
        }
      return true;
    }
 private:
  unsigned num_bit;
};


/** eoDetSingleBitFlip --> changes exactly k bits with checking for duplicate
\class eoDetSingleBitFlip eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoDetSingleBitFlip: public eoMonOp<Chrom>
{
 public:
  /**
   * (Default) Constructor.
   * @param _num_bit The number of bits to change
   * default is one - equivalent to eoOneBitFlip then
   */
  eoDetSingleBitFlip(const unsigned& _num_bit = 1): num_bit(_num_bit) {}

  /// The class name.
  virtual std::string className() const { return "eoDetSingleBitFlip"; }

  /**
   * Change num_bit bits.
   * @param chrom The cromosome which one bit is going to be changed.
   */
  bool operator()(Chrom& chrom)
    {
      std::vector< unsigned > selected;

      // check for duplicate
      for (unsigned k=0; k<num_bit; k++)
        {
	    unsigned temp;

	    do
		{
		    temp = eo::rng.random( chrom.size() );
		}
	    while ( find( selected.begin(), selected.end(), temp ) != selected.end() );

	    selected.push_back(temp);
        }

	for ( size_t i = 0; i < selected.size(); ++i )
	    {
		chrom[i] = !chrom[i];
	    }

      return true;
    }
 private:
  unsigned num_bit;
};


/** eoBitMutation --> classical mutation
\class eoBitMutation eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBitMutation: public eoMonOp<Chrom>
{
 public:
  /**
   * (Default) Constructor.
   * @param _rate Rate of mutation.
   * @param _normalize use rate/chrom.size if true
   */
  eoBitMutation(const double& _rate = 0.01, bool _normalize=false):
    rate(_rate), normalize(_normalize) {}

  /// The class name.
  virtual std::string className() const { return "eoBitMutation"; }

  /**
   * Mutate a chromosome.
   * @param chrom The chromosome to be mutated.
   */
  bool operator()(Chrom& chrom)
    {
      double actualRate = (normalize ? rate/chrom.size() : rate);
      bool changed_something = false;
      for (unsigned i = 0; i < chrom.size(); i++)
            if (eo::rng.flip(actualRate))
        {
                chrom[i] = !chrom[i];
            changed_something = true;
        }

        return changed_something;
    }

 private:
  double rate;
  bool normalize;                  // divide rate by chromSize
};


/** eoBitInversion: inverts the bits of the chromosome between an interval
\class eoBitInversion eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBitInversion: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  virtual std::string className() const { return "eoBitInversion"; }

  /**
   * Inverts a range of bits in a binary chromosome.
   * @param chrom The chromosome whos bits are going to be inverted (a range).
   */
  bool operator()(Chrom& chrom)
    {

      unsigned u1 = eo::rng.random(chrom.size() + 1) , u2;
      do u2 = eo::rng.random(chrom.size() + 1); while (u1 == u2);
      unsigned r1 = std::min(u1, u2), r2 = std::max(u1, u2);

      std::reverse(chrom.begin() + r1, chrom.begin() + r2);
      return true;
    }
};


/** eoBitNext --> next value when bitstring considered as binary value
\class eoBitNext eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBitNext: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  virtual std::string className() const { return "eoBitNext"; }

  /**
   * Change the bit std::string x to be x+1.
   * @param chrom The chromosome to be added one.
   */
  bool operator()(Chrom& chrom)
    {
      for (int i = chrom.size() - 1; i >= 0; i--)
        if (chrom[i])
          {
            chrom[i] = 0;
            continue;
          }
        else
          {
            chrom[i] = 1;
            break;
          }

    return true;
    }
};


/** eoBitPrev --> previous value when bitstring treated as binary value
\class eoBitPrev eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBitPrev: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  virtual std::string className() const { return "eoBitPrev"; }

  /**
   * Change the bit std::string x to be x-1.
   * @param chrom The chromosome to be substracted one.
   */
  bool operator()(Chrom& chrom)
    {
      for (int i = chrom.size() - 1; i >= 0; i--)
        if (chrom[i])
          {
            chrom[i] = 0;
            break;
          }
        else
          {
            chrom[i] = 1;
            continue;
          }

    return true;
    }
};


/** eo1PtBitXover --> classic 1-point crossover
\class eo1PtBitCrossover eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eo1PtBitXover: public eoQuadOp<Chrom>
{
 public:
  /// The class name.
  virtual std::string className() const { return "eo1PtBitXover"; }

  /**
   * 1-point crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  bool operator()(Chrom& chrom1, Chrom& chrom2)
    {
      unsigned site = eo::rng.random(std::min(chrom1.size(), chrom2.size()));

      if (!std::equal(chrom1.begin(), chrom1.begin()+site, chrom2.begin()))
      {

        std::swap_ranges(chrom1.begin(), chrom1.begin() + site, chrom2.begin());

        return true;
      }
      return false;
  }
};


/** eoUBitXover --> classic Uniform crossover
\class eoUBitXover eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoUBitXover: public eoQuadOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoUBitXover(const float& _preference = 0.5): preference(_preference)
    {
      if ( (_preference <= 0.0) || (_preference >= 1.0) )
        std::runtime_error("UxOver --> invalid preference");
    }
  /// The class name.
  virtual std::string className() const { return "eoUBitXover"; }

  /**
   * Uniform crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   *    std::runtime_error if sizes don't match
   */
  bool operator()(Chrom& chrom1, Chrom& chrom2)
    {
      if ( chrom1.size() != chrom2.size())
            std::runtime_error("UxOver --> chromosomes sizes don't match" );
      bool changed = false;
      for (unsigned int i=0; i<chrom1.size(); i++)
        {
          if (chrom1[i] != chrom2[i] && eo::rng.flip(preference))
            {
              bool tmp = chrom1[i];
              chrom1[i]=chrom2[i];
              chrom2[i] = tmp;
              changed = true;
            }
        }
    return changed;
  }
    private:
      float preference;
};


/** eoNPtsBitXover --> n-point crossover
\class eoNPtsBitXover eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/
template<class Chrom> class eoNPtsBitXover : public eoQuadOp<Chrom>
{
public:

    /** (Default) Constructor. */
    eoNPtsBitXover(const unsigned& _num_points = 2) : num_points(_num_points)
        {
            if (num_points < 1)
                std::runtime_error("NxOver --> invalid number of points");
        }

    /** The class name */
    virtual std::string className() const { return "eoNPtsBitXover"; }

    /** n-point crossover for binary chromosomes.

    @param chrom1 The first chromosome.
    @param chrom2 The first chromosome.
    */
    bool operator()(Chrom& chrom1, Chrom& chrom2) {
        unsigned max_size(std::min(chrom1.size(), chrom2.size()));
        unsigned max_points(std::min(max_size - 1, num_points));
        std::vector<bool> points(max_size, false);

        // select ranges of bits to swap
        do {
            unsigned bit(eo::rng.random(max_size));
            if(points[bit])
                continue;
            else {
                points[bit] = true;
                --max_points;
            }
        } while(max_points);

        // swap bits between chromosomes
        bool change(false);
        for (unsigned bit = 1; bit < points.size(); bit++) {
            if (points[bit])
                change = !change;
            if (change) {
                typename Chrom::AtomType tmp = chrom1[bit];
                chrom1[bit] = chrom2[bit];
                chrom2[bit] = tmp;
            }
        }
        return true;
    }

private:

    /** @todo Document this data member */
    unsigned num_points;
};



/** eoBitGxOver --> Npts crossover when bistd::string considered
                    as a std::string of binary-encoded genes (exchanges genes)
Is anybody still using it apart from historians ??? :-)
\class eoBitGxOver eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBitGxOver: public eoQuadOp<Chrom>
{
 public:
  /// Constructor.
  eoBitGxOver(const unsigned _gene_size, const unsigned _num_points = 2):
    gene_size(_gene_size), num_points(_num_points)
    {
      if (gene_size < 1)
        std::runtime_error("GxOver --> invalid gene size");
      if (num_points < 1)
        std::runtime_error("GxOver --> invalid number of points");
    }

  /// The class name
  virtual std::string className() const { return "eoBitGxOver"; }

  /**
   * Gene crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  bool operator()(Chrom& chrom1, Chrom& chrom2)
    {
      unsigned max_genes = std::min(chrom1.size(), chrom2.size()) / gene_size;
      unsigned cut_genes = std::min(max_genes, num_points);

      std::vector<bool> points(max_genes, false);

      // selects genes to swap
      do {
        unsigned bit = eo::rng.random(max_genes);
        if (points[bit])
          continue;
        else
          {
            points[bit] = true;
            cut_genes--;
          }
      } while (cut_genes);

      // swaps genes
      for (unsigned i = 0; i < points.size(); i++)
        if (points[i])
          std::swap_ranges(chrom1.begin() + i * gene_size,
                      chrom1.begin() + i * gene_size + gene_size,
                      chrom2.begin() + i * gene_size);

    return true;
  }

 private:
  unsigned gene_size;
  unsigned num_points;
};



//-----------------------------------------------------------------------------
//@}
#endif
