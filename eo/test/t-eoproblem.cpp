//-----------------------------------------------------------------------------
// t-eoproblem.cpp
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#include <time.h>    // time
#include <math.h>    // fabs
#include <iostream>  // std::cout
#include <eo>        // eoVector, eoProblem

//-----------------------------------------------------------------------------

typedef eoVector<float, float> Chrom;

//-----------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const Chrom& chrom)
{
  copy(chrom.begin(), chrom.end(), std::ostream_iterator<int>(os));
  return os;
}

//-----------------------------------------------------------------------------

class Easy//: public eoProblem<Chrom>
{
public:
  static const unsigned size;
  
  float operator()(const Chrom& chrom)
    {
      return 1.0 / (fabs(chrom[0]) + 1.0);
    }
};
const unsigned Easy::size = 1;

//-----------------------------------------------------------------------------

int main()
{
  Easy easy;
  Chrom chrom(Easy::size);
  
  srand(time(0));
  
  chrom[0] = ((float)rand()) / ((float)RAND_MAX);
  chrom.fitness(easy(chrom));
  
  std::cout << "chrom = " << chrom << std::endl
       << "chrom.fitness() = " << chrom.fitness() << std::endl;

  return 0;
}

//-----------------------------------------------------------------------------
