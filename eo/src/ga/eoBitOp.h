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
 */
//-----------------------------------------------------------------------------
// MS 17/10/2000
// Added the uniform crossover - which, for some reasons, had dissapeared!
// Added the eoDetBitFlip, which flips exactly num_bit bits
// Aslo added the above standard header

// I also want to start the discussion about the "gene" crossover.
// I think the word "gene" is not appropriate: if real numbers are coded in 
// binary format, then a "gene" is a bit, and that's it
// if you want to exchange real number per se, then use real coding
//
// Because all crossover operators here except that Gene crossover 
// ARE generic, i.e. appky to any vertor of something.

// Note that for mutations, if instead of       
//             chrom[i] = (chrom[i]) ? false : true;
// we were calling something like
//             specific_mutate(chrom[i])
// all mutation would also be generic ... except those eoBinNext and eoBinPrev

// If anybody reads this and want to change that (I'm also testing to see
// if someone ever reads the headers :-), drop me a mail
// Marc (Marc.Schoenauer@polytechnique.fr)
//-----------------------------------------------------------------------------
// eoBitOp.h
//-----------------------------------------------------------------------------

#ifndef eoBitOp_h
#define eoBitOp_h

//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoRNG.h>
#include <eoInit.h>       // eoMonOp
#include <ga/eoBin.h>


/** eoBinBitFlip --> changes 1 bit  
\class eoBinBitFlip eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinBitFlip: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinBitFlip"; }
  
  /**
   * Change one bit.
   * @param chrom The cromosome which one bit is going to be changed.
   */
  void operator()(Chrom& chrom) 
    {
      chrom.invalidate();
      unsigned i = rng.random(chrom.size());
      chrom[i] = (chrom[i]) ? false : true;
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
   * default is one - equivalent to eoBinBitFlip then
   */
  eoDetBitFlip(const unsigned& _num_bit = 1): num_bit(_num_bit) {}

  /// The class name.
  string className() const { return "eoDetBitFlip"; }
  
  /**
   * Change num_bit bits.
   * @param chrom The cromosome which one bit is going to be changed.
   */
  void operator()(Chrom& chrom) 
    {
      chrom.invalidate();
      // does not check for duplicate: if someone volunteers ....
      for (unsigned k=0; k<num_bit; k++)
	{
	  unsigned i = rng.random(chrom.size());
	  chrom[i] = (chrom[i]) ? false : true;
	}
    }
 private:
  unsigned num_bit;
};


/** eoBinMutation --> classical mutation 
\class eoBinMutation eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinMutation: public eoMonOp<Chrom>
{
 public:
  /**
   * (Default) Constructor.
   * @param _rate Rate of mutation.
   */
  eoBinMutation(const double& _rate = 0.01): rate(_rate) {}

  /// The class name.
  string className() const { return "eoBinMutation"; }

  /**
   * Mutate a chromosome.
   * @param chrom The chromosome to be mutated.
   */
  void operator()(Chrom& chrom) 
    {
      bool changed_something = false;
      for (unsigned i = 0; i < chrom.size(); i++)
	    if (rng.flip(rate))
        {
	        chrom[i] = !chrom[i];
            changed_something = true;
        }

        if (changed_something)
            chrom.invalidate();
    }
  
 private:
  double rate;
};


/** eoBinInversion: inverts the bits of the chromosome between an interval 
\class eoBinInversion eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinInversion: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinInversion"; }

  /**
   * Inverts a range of bits in a binary chromosome.
   * @param chrom The chromosome whos bits are going to be inverted (a range).
   */
  void operator()(Chrom& chrom) 
    {
      
      unsigned u1 = rng.random(chrom.size() + 1) , u2;
      do u2 = rng.random(chrom.size() + 1); while (u1 == u2);
      unsigned r1 = min(u1, u2), r2 = max(u1, u2);
      
      reverse(chrom.begin() + r1, chrom.begin() + r2);
      chrom.invalidate();
    }
};


/** eoBinNext --> next binary value 
\class eoBinNext eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinNext: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinNext"; }
  
  /**
   * Change the bit string x to be x+1.
   * @param chrom The chromosome to be added one.
   */
  void operator()(Chrom& chrom) 
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

    chrom.invalidate();
    }
};


/** eoBinPrev --> previous binary value 
\class eoBinPrev eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinPrev: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinPrev"; }
  
  /**
   * Change the bit string x to be x-1.
   * @param chrom The chromosome to be substracted one.
   */
  void operator()(Chrom& chrom) 
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

    chrom.invalidate();
    }
};
  

/** eoBinCrossover --> classic 1-point crossover 
\class eoBinCrossover eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinCrossover: public eoQuadraticOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinCrossover"; }

  /**
   * 1-point crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  void operator()(Chrom& chrom1, Chrom& chrom2) 
    {
      unsigned site = rng.random(min(chrom1.size(), chrom2.size()));

      if (!std::equal(chrom1.begin(), chrom1.begin()+site, chrom2.begin()))
      {

        swap_ranges(chrom1.begin(), chrom1.begin() + site, chrom2.begin());
  
        chrom1.invalidate();
        chrom2.invalidate();
      }
  }
};
  

/** eoBinUxOver --> classic Uniform crossover 
\class eoBinNxOver eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinUxOver: public eoQuadraticOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoBinUxOver(const float& _preference = 0.5): preference(_preference)
    { 
      if ( (_preference <= 0.0) || (_preference >= 1.0) )
	runtime_error("UxOver --> invalid preference");
    }
  /// The class name.
  string className() const { return "eoBinUxOver"; }

  /**
   * Uniform crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   *    @runtime_error if sizes don't match
   */
  void operator()(Chrom& chrom1, Chrom& chrom2) 
    {
      if ( chrom1.size() != chrom2.size()) 
	    runtime_error("UxOver --> chromosomes sizes don't match" ); 
      bool changed = false;
      for (unsigned int i=0; i<chrom1.size(); i++)
	{
	  if (rng.flip(preference))
	    {
	      bool tmp = chrom1[i];
	      chrom1[i]=chrom2[i];
	      chrom2[i] = tmp;
	      changed = true;
	    }
	}
      if (changed)
	  {
	    chrom1.invalidate();
	    chrom2.invalidate();
	  }
    }
    private:
      float preference;
};
  

/** eoBinNxOver --> n-point crossover 
\class eoBinNxOver eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinNxOver: public eoQuadraticOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoBinNxOver(const unsigned& _num_points = 2): num_points(_num_points)
    { 
      if (num_points < 1)
	runtime_error("NxOver --> invalid number of points");
    }
  
  /// The class name.
  string className() const { return "eoBinNxOver"; }
  
  /**
   * n-point crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  void operator()(Chrom& chrom1, Chrom& chrom2) 
    {
      unsigned max_size = min(chrom1.size(), chrom2.size());
      unsigned max_points = min(max_size - 1, num_points);
      
      vector<bool> points(max_size, false);
      
      // select ranges of bits to swap
      do {
	unsigned bit = rng.random(max_size) + 1; 
	if (points[bit])
	  continue;
	else
	  {
	    points[bit] = true;
	    max_points--;
	  }
      } while (max_points);
      
      
      // swap bits between chromosomes
      bool change = false;
      for (unsigned bit = 1; bit < points.size(); bit++)
	{
	  if (points[bit])
	    change = !change;
	  
	  if (change)
	    swap(chrom1[bit], chrom2[bit]);
	}

      chrom1.invalidate();
      chrom2.invalidate();
    }
    
 private:
  unsigned num_points;
};



/** eoBinGxOver --> gene crossover 
\class eoBinGxOver eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinGxOver: public eoQuadraticOp<Chrom>
{
 public:
  /// Constructor.
  eoBinGxOver(const unsigned _gene_size, const unsigned _num_points = 2): 
    gene_size(_gene_size), num_points(_num_points)
    {  
      if (gene_size < 1)
	runtime_error("GxOver --> invalid gene size");
      if (num_points < 1)
	runtime_error("GxOver --> invalid number of points");
    }
  
  /// The class name
  string className() const { return "eoBinGxOver"; }
  
  /**
   * Gene crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  void operator()(Chrom& chrom1, Chrom& chrom2) 
    {
      unsigned max_genes = min(chrom1.size(), chrom2.size()) / gene_size;
      unsigned cut_genes = min(max_genes, num_points);
      
      vector<bool> points(max_genes, false);
      
      // selects genes to swap
      do {
	unsigned bit = rng.random(max_genes); 
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
	  swap_ranges(chrom1.begin() + i * gene_size, 
		      chrom1.begin() + i * gene_size + gene_size, 
		      chrom2.begin() + i * gene_size);
  
    chrom1.invalidate();
    chrom2.invalidate();
  }
  
 private:
  unsigned gene_size;
  unsigned num_points;
};



//-----------------------------------------------------------------------------
//@}
#endif eoBitOp_h

