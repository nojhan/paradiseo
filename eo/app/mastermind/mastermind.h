//-----------------------------------------------------------------------------
// mastermind.h
//-----------------------------------------------------------------------------

#ifndef mastermind_h
#define mastermind_h

//-----------------------------------------------------------------------------

#include <stdlib.h>               // exit EXIT_FAILURE
#include <eoFixedLength.h>        // eoFixedLength
#include <eoOp.h>                 // eoMonOp eoQuadraticOp
#include <eoInit.h>               // eoInit
#include <utils/rnd_generators.h> // uniform_generator

//-----------------------------------------------------------------------------
// phenotype
//-----------------------------------------------------------------------------

typedef float phenotype;

//-----------------------------------------------------------------------------
// genotype
//-----------------------------------------------------------------------------

typedef vector<int> genotype;

//-----------------------------------------------------------------------------
// Chrom
//-----------------------------------------------------------------------------

typedef eoFixedLength<phenotype, int> Chrom;

//-----------------------------------------------------------------------------
// eoChromEvaluator
//-----------------------------------------------------------------------------

const unsigned default_size = 8;
const string default_solution = "01234567";

Chrom solution;
unsigned num_colors;

void init_eoChromEvaluator(const unsigned& c, const unsigned& l, string s)
{
  num_colors = c;

  // generate a random solution
  if (s == default_solution || s.size() != l)
    {
      uniform_generator<char> color('0', static_cast<char>('0' + num_colors));
      s.resize(l);
      generate(s.begin(), s.end(), color);
    }

  // check solution
  for (unsigned i = 0; i < solution.size(); ++i)
    if (solution[i] >= num_colors)
      {
	cerr << "too high color number found!" << endl;
	exit(EXIT_FAILURE);
      }
  
  solution.resize(s.size());
  for (unsigned i = 0; i < solution.size(); ++i)
    solution[i] = s[i] - '0';
}

const unsigned points_per_black = 3, points_per_white = 1;

phenotype eoChromEvaluator(const Chrom& chrom)
{
  Chrom tmp = solution;
  unsigned black = 0, white = 0;

  // look for blacks
  for (unsigned i = 0; i < chrom.size(); ++i)
    if (chrom[i] == tmp[i])
      {
	++black;
	tmp[i] = -1;
      }
  
  // look for whites
  for (unsigned i = 0; i < chrom.size(); ++i)
    for (unsigned j = 0; j < tmp.size(); ++j)
      if (chrom[i] == tmp[j])
	{
	  ++white;
	  tmp[j] = -1;
	  break;
	}

  //  return black * points_per_black + white * points_per_white;
  return black * chrom.size() + white;
};

//-----------------------------------------------------------------------------
// eoChromInit
//-----------------------------------------------------------------------------

class eoInitChrom: public eoInit<Chrom>
{
public:
  void operator()(Chrom& chrom)
  {
    uniform_generator<int> color(0, num_colors);
    chrom.resize(solution.size());
    generate(chrom.begin(), chrom.end(), color);
    chrom.invalidate();
  }
};

//-----------------------------------------------------------------------------
// eoChromMutation
//-----------------------------------------------------------------------------

class eoChromMutation: public eoMonOp<Chrom>
{
  // two changes in one mutation :(
  void operator()(Chrom& chrom)
  {
    uniform_generator<unsigned> position(0, chrom.size());

    // random gene change
    uniform_generator<int> color(0, num_colors);
    chrom[position()] = color();
    
    // random gene swap
    swap(chrom[position()], chrom[position()]);

    chrom.invalidate();
  }
};

//-----------------------------------------------------------------------------
// eoChromXover
//-----------------------------------------------------------------------------

class eoChromXover: public eoQuadraticOp<Chrom>
{
public:
  void operator()(Chrom& chrom1, Chrom& chrom2)
  {
    uniform_generator<unsigned> position(0, chrom1.size());
    swap_ranges(chrom1.begin(), chrom1.begin() + position(), chrom2.begin());
    chrom1.invalidate();
    chrom2.invalidate();
  }
};

//-----------------------------------------------------------------------------

#endif // mastermind_h

// Local Variables: 
// mode:C++ 
// End:
