//-----------------------------------------------------------------------------
// mastermind.h
//-----------------------------------------------------------------------------

#ifndef mastermind_h
#define mastermind_h

//-----------------------------------------------------------------------------

#include <stdlib.h>               // exit EXIT_FAILURE
#include <eoVector.h>             // eoVectorLength
#include <eoOp.h>                 // eoMonOp eoQuadraticOp
#include <eoInit.h>               // eoInit
#include "utils/rnd_generators.h" // uniform_generator

//-----------------------------------------------------------------------------
// phenotype
//-----------------------------------------------------------------------------

typedef float phenotype;

//-----------------------------------------------------------------------------
// genotype
//-----------------------------------------------------------------------------

typedef std::vector<int> genotype;

//-----------------------------------------------------------------------------
// Chrom
//-----------------------------------------------------------------------------

typedef eoVector<phenotype, int> Chrom;

//-----------------------------------------------------------------------------
// eoChromEvaluator
//-----------------------------------------------------------------------------

// const unsigned points_per_black = 3, points_per_white = 1;
Chrom solution;

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
}

const unsigned default_length = 8;
const unsigned default_colors = 8;
const std::string default_solution = "01234567";


unsigned num_colors;

void init_eoChromEvaluator(const unsigned& c, const unsigned& l, std::string s)
{
  num_colors = c;

  // check consistency between parameters
  if (s != default_solution)
    {
      // check length
      if (l != default_length && s.size() != l)
	{
	  std::cerr << "solution length != length" << std::endl;
	  exit(EXIT_FAILURE);
	}

      // check number of colors
      if ((c != default_colors) && (c < unsigned(*max_element(s.begin(), s.end()) - '0')))
	{
	  std::cerr << "too high color number found!" << std::endl;
	  exit(EXIT_FAILURE);
	}
    }
  else
    if (l != default_length || c != default_colors )
      // generate a random solution
      if(num_colors <= 10)
	{
	  uniform_generator<char> color('0', static_cast<char>('0' + c));
	  s.resize(l);
	  generate(s.begin(), s.end(), color);
	}

  // put the solution parameter on the solution chromosome
  if (num_colors <= 10)
    {
      solution.resize(s.size());
      for (unsigned i = 0; i < solution.size(); ++i)
	solution[i] = s[i] - '0';
    }
  else
    {
      solution.resize(l);
      uniform_generator<int> color(0, num_colors);
      generate(solution.begin(), solution.end(), color);
    }

  solution.fitness(eoChromEvaluator(solution));
}

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
  // many operators in one :(
  bool operator()(Chrom& chrom)
  {
    uniform_generator<unsigned> what(0, 2);
    uniform_generator<unsigned> position(0, chrom.size());

    switch(what())
      {
      case 0:
	{
	  // mutation
	  uniform_generator<int> color(0, num_colors);
	  chrom[position()] = color();
	  break;
	}
      case 1:
	{
	  // transposition
	  std::swap(chrom[position()], chrom[position()]);
	  break;
	}
      default:
	{
	  std::cerr << "unknown operator!" << std::endl;
	  exit(EXIT_FAILURE);
	  break;
	}
      }

    return true;
  }
};

//-----------------------------------------------------------------------------
// eoChromXover
//-----------------------------------------------------------------------------

class eoChromXover: public eoQuadOp<Chrom>
{
public:
  bool operator()(Chrom& chrom1, Chrom& chrom2)
  {
    uniform_generator<unsigned> position(0, chrom1.size());
    swap_ranges(chrom1.begin(), chrom1.begin() + position(), chrom2.begin());
    return true;
  }
};

//-----------------------------------------------------------------------------

#endif // mastermind_h

// Local Variables:
// mode:C++
// End:
