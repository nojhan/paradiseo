//-----------------------------------------------------------------------------
// eoBitOp.h
//-----------------------------------------------------------------------------

#ifndef eoBitOp_h
#define eoBitOp_h

//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoRNG.h>
#include <ga/eoBin.h>      // eoBin
#include <eoOp.h>       // eoMonOp


/** @name BitWise Genetic operators

\class eoBinRandom eoBitOp.h ga/eoBitOp.h
\ingroup bitstring

Even as these operators might seem general, they are particular versions of genetic
operators useful only for binary operators. As any set of genetic operators, it must
have a factory that knows how to build them from a description
@author GeNeura Team
@version 0.1
@see eoBin
@see eoBitOpFactory
*/

//@{

/** eoBinRandom --> mofify a chromosome in a random way */

template<class Chrom> class eoBinRandom: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinRandom"; }

  /**
   * Randomizes a cromosome.
   * @param chrom The cromosome to be randomize.
   */
  void operator()(Chrom& chrom) const
    {
      for (unsigned i = 0; i < chrom.size(); i++)
	chrom[i] = rng.flip(0.5) ? false : true;
    }
};


/** eoBinBitFlip --> changes a bit 
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
  void operator()(Chrom& chrom) const
    {
      unsigned i = rng.random(chrom.size());
      chrom[i] = (chrom[i]) ? false : true;
    }
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
  void operator()(Chrom& chrom) const
    {
      for (unsigned i = 0; i < chrom.size(); i++)
	if (rng.flip(rate))
	  chrom[i] = !chrom[i];
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
  void operator()(Chrom& chrom) const
    {
      
      unsigned u1 = rng.random(chrom.size() + 1) , u2;
      do u2 = rng.random(chrom.size() + 1); while (u1 == u2);
      unsigned r1 = min(u1, u2), r2 = max(u1, u2);
      
      reverse(chrom.begin() + r1, chrom.begin() + r2);
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
  void operator()(Chrom& chrom) const
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
  void operator()(Chrom& chrom) const
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
    }
};
  

/** eoBinCrossover --> classic 2-point crossover 
\class eoBinCrossover eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinCrossover: public eoQuadraticOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinCrossover"; }

  /**
   * 2-point crossover for binary chromosomes.
   * @param chrom1 The first chromosome.
   * @param chrom2 The first chromosome.
   */
  void operator()(Chrom& chrom1, Chrom& chrom2) const
    {
      swap_ranges(chrom1.begin(), chrom1.begin() + rng.random(min(chrom1.size(), chrom2.size())), chrom2.begin());
    }
};
  

/** eoBinNxOver --> n-point crossover 
\class eoBinNxOver eoBitOp.h ga/eoBitOp.h
\ingroup bitstring
*/

template<class Chrom> class eoBinNxOver: public eoQuadraticOp<Chrom>
{
 public:
  /// (Defualt) Constructor.
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
  void operator()(Chrom& chrom1, Chrom& chrom2) const
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
  void operator()(Chrom& chrom1, Chrom& chrom2) const
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
    }
  
 private:
  unsigned gene_size;
  unsigned num_points;
};



//-----------------------------------------------------------------------------
//@}
#endif eoBitOp_h

