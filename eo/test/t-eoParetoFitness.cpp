#include "eoParetoFitness.h"

class MinimizingTraits : public eoParetoFitnessTraits
{
public :

  static bool maximizing(int) { return false; }
};

int main()
{
  typedef eoParetoFitness<> MaxFitness;
  typedef eoParetoFitness<MinimizingTraits> MinFitness;

  MaxFitness f0;
  f0[0] = 0.0;
  f0[1] = 1.0;

  MaxFitness f1;
  f1[0] = 1.0;
  f1[1] = 0.0;

  MaxFitness f2;
  f2[0] = 0.0;
  f2[1] = 0.5;

  // now f0 should dominate f2;

  if (!f0.dominates(f2))
  {
    cout << f2 << " not dominated by " << f0;
    throw;
  }

  // f0 and f1 should not dominate each other

  if (f0.dominates(f1) || f1.dominates(f0))
  {
    cout << f0 << " and " << f1 << " dominate";
    throw;
  }

  if (! (f0 == f0))
  {
    cout << "f0 == f0 failed" << endl;
    throw;
  }

  // test ctors and such
  MaxFitness f3 = f0;
  f3[0] += 1e-9;

  // test tolerance
  assert(f3 == f0);

  MinFitness m0;
  MinFitness m1;
  MinFitness m2;
  MinFitness m3;

  m0[0] = 0.0;
  m0[1] = 1.0;

  m1[0] = 1.0;
  m1[1] = 0.0;

  m2[0] = 0.0;
  m2[1] = 0.5;

  m3[0] = 0.5;
  m3[1] = 0.5;

  //m2 should dominate m0
  assert(m2.dominates(m0));

  assert(!m1.dominates(m0));
  assert(!m0.dominates(m1));
  assert(!m0.dominates(m2)); // (m2 < m0));
  assert(m2.dominates(m3)); //m3 < m2);
  assert(!m3.dominates(m2)); // (m2 < m3));
  assert(m2.dominates(m3)); //m2 > m3);
}