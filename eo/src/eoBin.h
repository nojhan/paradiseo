//-----------------------------------------------------------------------------
// eoBin.h
//-----------------------------------------------------------------------------

#ifndef eoBin_h
#define eoBin_h

//-----------------------------------------------------------------------------

#include <iostream>     // ostream, istream
#include <bvector.h>    // bit_vector
#include <string>       // string
#include <EO.h>         // EO
#include <eoOp.h>       // eoMonOp
#include <eoUniform.h>  // eoUniform
#include <eoRnd.h>      // eoRnd

//-----------------------------------------------------------------------------

template <class F> class eoBin: public EO<F>, public bit_vector
{
 public:
  typedef bool Type;

  /// (Default) Constructor.
  /// @param size Size of the binary string.
  eoBin(const unsigned& size = 0, const bool& value = false): 
    bit_vector(size, value) {}

  /// (Default) Constructor.
  /// @param size Size of the binary string.
  eoBin(const unsigned& size, const eoRnd<Type>& rnd): bit_vector(size) 
    {
      generate(begin(), end(), rnd);
    }
  
  /// Constructor from istream.
  /// @param is The istream to read from.
  eoBin(istrstream& is)
    { 
      readFrom(is); 
    }
  
  /// My class name.
  string className() const 
    { 
      return "eoBin"; 
    }
  
  /// To print me on a stream.
  /// @param os The ostream.
  void printOn(ostream& os) const
    {
      copy(begin(), end(), ostream_iterator<bool>(os));
    }
  
  /// To read me from a stream.
  /// @param is The istream.
  void readFrom(istream& is)
    {
      string bits;
      is >> bits;
      if (is)
	{
	  resize(bits.size());
	  transform(bits.begin(), bits.end(), begin(), 
		    bind2nd(equal_to<char>(), '1'));
	}
    }
};

//-----------------------------------------------------------------------------
// eoBinRandom --> mofify a chromosome in a random way
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinRandom: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinRandom"; }

  ///
  void operator()(Chrom& chrom) const
    {
      eoUniform<float> uniform(0.0, 1.0);
      for (unsigned i = 0; i < chrom.size(); i++)
	chrom[i] = (uniform() < 0.5) ? false : true;
    }
};


//-----------------------------------------------------------------------------
// eoBinBitFlip --> chages a bit
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinBitFlip: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinBitFlip"; }
  
  ///
  void eoBinBitFlip::operator()(Chrom& chrom) const
    {
      eoUniform<unsigned> uniform(0, chrom.size());
      unsigned pos = uniform();
      chrom[pos] = !chrom[pos];
    }
};

//-----------------------------------------------------------------------------
// eoBinMutation --> classical mutation
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinMutation: public eoMonOp<Chrom>
{
 public:
  /// (Default) Constructor.
  eoBinMutation(const double& _rate = 0.01): rate(_rate), uniform(0.0, 1.0) {}

  /// The class name.
  string className() const { return "eoBinMutation"; }

  ///
  void operator()(Chrom& chrom) const
    {
      for (unsigned i = 0; i < chrom.size(); i++)
	if (uniform() < rate)
	  chrom[i] = !chrom[i];
    }
  
 private:
  double rate;
  mutable eoUniform<float> uniform;
};

//-----------------------------------------------------------------------------
// eoBinInversion --> inverts the bits of the chromosome between an interval
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinInversion: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinInversion"; }

  ///  
  void operator()(Chrom& chrom) const
    {
      eoUniform<unsigned> uniform(0, chrom.size() + 1);
      
      unsigned u1 = uniform(), u2;
      do u2 = uniform(); while (u1 == u2);
      unsigned r1 = min(u1, u2), r2 = max(u1, u2);
      
      reverse(chrom.begin() + r1, chrom.begin() + r2);
    }
};

//-----------------------------------------------------------------------------
// eoBinNext --> next binary value
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinNext: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinNext"; }
  
  ///
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

//-----------------------------------------------------------------------------
// eoBinPrev --> previos binary value
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinPrev: public eoMonOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinPrev"; }

  ///
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

//-----------------------------------------------------------------------------
// eoBinCrossover --> classic crossover
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinCrossover: public eoBinOp<Chrom>
{
 public:
  /// The class name.
  string className() const { return "eoBinCrossover"; }

  void operator()(Chrom& chrom1, Chrom& chrom2) const
    {
      eoUniform<unsigned> uniform(1, min(chrom1.size(), chrom2.size()));
      swap_ranges(chrom1.begin(), chrom1.begin() + uniform(), chrom2.begin());
    }
};
  
//-----------------------------------------------------------------------------
// eoBinNxOver --> n-point crossover
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinNxOver: public eoBinOp<Chrom>
{
 public:
  /// (Defualt) Constructor.
  eoBinNxOver(const unsigned& _num_points = 2): num_points(_num_points)
    { 
      if (num_points < 1)
	{
	  cerr << "NxOver --> invalid number of points " << num_points << endl;
	  exit(EXIT_FAILURE);
	}
    }

  /// The class name.
  string className() const { return "eoBinNxOver"; }
  
  ///
  void operator()(Chrom& chrom1, Chrom& chrom2) const
    {
      unsigned max_size = min(chrom1.size(), chrom2.size());
      unsigned max_points = min(max_size - 1, num_points);
      
      vector<bool> points(max_size, false);
      eoUniform<unsigned> uniform(1, max_size);
      
      // select ranges of bits to swap
      do {
	unsigned bit = uniform();
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

//-----------------------------------------------------------------------------
// eoBinGxOver --> gene crossover
//-----------------------------------------------------------------------------

template<class Chrom> class eoBinGxOver: public eoBinOp<Chrom>
{
 public:
  /// Constructor.
  eoBinGxOver(const unsigned _gene_size, const unsigned _num_points = 2): 
    gene_size(_gene_size), num_points(_num_points)
    {  
      if (gene_size < 1)
	{
	  cerr << "GxOver --> invalid gene size " << gene_size << endl;
	  exit(EXIT_FAILURE);
	}
      if (num_points < 1)
	{
	  cerr << "GxOver --> invalid number of points " << num_points << endl;
	  exit(EXIT_FAILURE);
	}
    }
  
  /// The class name
  string className() const { return "eoBinGxOver"; }

  ///
  void operator()(Chrom& chrom1, Chrom& chrom2) const
    {
      unsigned max_genes = min(chrom1.size(), chrom2.size()) / gene_size;
      unsigned cut_genes = min(max_genes, num_points);
      
      vector<bool> points(max_genes, false);
      eoUniform<unsigned> uniform(0, max_genes);
      
      // selects genes to swap
      do {
	unsigned bit = uniform();
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
// eoBinUxOver --> uniform crossover
//-----------------------------------------------------------------------------
      
template<class Chrom> class eoBinUxOver: public eoBinOp<Chrom>
{
 public:
  /// Constructor.
  eoBinUxOver(const float _rate): rate(_rate)
    { 
      if (rate < 0 || rate > 1)
	{
	  cerr << "UxOver --> invalid rate " << rate << endl;
	  exit(EXIT_FAILURE);
	}
    }
  
  /// The class name.
  string className() const { return "eoBinUxOver"; }

  ///
  void operator()(Chrom& chrom1, Chrom& chrom2) const
    {
      unsigned min_size = min(chrom1.size(), chrom2.size());
      eoUniform<float> uniform(0, 1);
      
      for (unsigned bit = 0; bit < min_size; bit++)
	if (uniform() < rate)
	  swap(chrom1[bit], chrom2[bit]);
    }
  
 public:
  float rate;
};

//-----------------------------------------------------------------------------


#endif eoBin_h
