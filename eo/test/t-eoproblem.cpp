//-----------------------------------------------------------------------------
// t-eoproblem.cpp
// (c) GeNeura Team 1998
//-----------------------------------------------------------------------------

#include <time.h>    // time
#include <math.h>    // fabs
#include <iostream>  // cout
#include <eo>        // eoVector, eoProblem

//-----------------------------------------------------------------------------

typedef eoVector<float, float> Chrom;

//-----------------------------------------------------------------------------

ostream& operator<<(ostream& os, const Chrom& chrom)
{
  copy(chrom.begin(), chrom.end(), ostream_iterator<int>(os));
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

main()
{
  Easy easy;
  Chrom chrom(Easy::size);
  
  srand(time(0));
  
  chrom[0] = ((float)rand()) / ((float)RAND_MAX);
  chrom.fitness(easy(chrom));
  
  cout << "chrom = " << chrom << endl
       << "chrom.fitness() = " << chrom.fitness() << endl;
}

//-----------------------------------------------------------------------------
