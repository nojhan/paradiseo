// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// t-eoParetoFitness.cpp
// (c) Maarten Keijzer
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

    Contact: mak@dhi.dk

30/01/02 - MS - Added the eoVariableParetoTraits - and the compare Fn
 */
//-----------------------------------------------------------------------------

#include <cassert>
#include <iostream>

#include "eoParetoFitness.h"

using namespace std;

/** test program for Pareto Fitness */

class MinimizingTraits : public eoParetoFitnessTraits
{
public :

  static bool maximizing(int) { return false; }
};

template <class F>
void compare(F & _eo1, F & _eo2)
{
  if (_eo1.dominates(_eo2))
    std::cout << _eo1 << " dominates " << _eo2 << std::endl;
  else if (_eo2.dominates(_eo1))
    std::cout << _eo2 << " dominates " << _eo1 << std::endl;
  else
    std::cout << "None of " << _eo1 << " and " << _eo2 << "dominates the other" << std::endl;
  return;
}

int main()
{
  typedef eoParetoFitness<> MaxFitness;
  typedef eoParetoFitness<MinimizingTraits> MinFitness;

  typedef eoParetoFitness<eoVariableParetoTraits> VarFitness;

  try{

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
    std::cout << f2 << " not dominated by " << f0;
    throw;
  }

  // f0 and f1 should not dominate each other

  if (f0.dominates(f1) || f1.dominates(f0))
  {
    std::cout << f0 << " and " << f1 << " dominate";
    throw;
  }

  if (! (f0 == f0))
  {
    std::cout << "f0 == f0 failed" << std::endl;
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


  //////////////////////////////////////////
  // now the run-time set-able number of objectives
  ////////////////////////////////////////////

  std::cout << "On y va" << std::endl;


  // setup fitness WARNING do not try to allocate any EO before that (runtime error)
  vector<bool> b(2, true);
  b[0]=true; 
  b[1]=false;
  VarFitness::setUp(2, b);
  std::cout << "\nMAXimizing on Obj 0 and MINimizing on Obj 1\n";

  VarFitness mv0;
  VarFitness mv1;
  VarFitness mv2;
  VarFitness mv3;

  mv0[0] = 0.0;
  mv0[1] = 1.0;

  mv1[0] = 1.0;
  mv1[1] = 0.0;

  mv2[0] = 0.0;
  mv2[1] = 0.5;

  mv3[0] = 0.5;
  mv3[1] = 0.5;

  compare <VarFitness>(mv0,mv1);
  compare <VarFitness>(mv0,mv2);
  compare <VarFitness>(mv0,mv3);
  compare <VarFitness>(mv1,mv2);
  compare <VarFitness>(mv1,mv3);
  compare <VarFitness>(mv2,mv3);

  std::cout << "\nChanging now the min <-> max\n";
  b[0]=false; 
  b[1]=true;
  VarFitness::setUp(2, b);
  std::cout << "\nMINimizing on Obj 0 and MAXimizing on Obj 1\n";
  compare <VarFitness>(mv0,mv1);
  compare <VarFitness>(mv0,mv2);
  compare <VarFitness>(mv0,mv3);
  compare <VarFitness>(mv1,mv2);
  compare <VarFitness>(mv1,mv3);
  compare <VarFitness>(mv2,mv3);

  std::cout << "\nTesting WARNING\n";
  b.resize(3);
  b[0]=false; 
  b[1]=true;
  b[2]=true;
  VarFitness::setUp(3, b);

  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

}
