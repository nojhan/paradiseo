/*
  <t-eoPartiallyMappedXover.cpp>
*/

#include <eo>
#include <eoInt.h>

#include <eoPartiallyMappedXover.h>

#include <cstdlib>
#include <cassert>

//-----------------------------------------------------------------------------

typedef eoInt<int> Solution;

int main() {

  std::cout << "[t-eoPartiallyMappedXover] => START" << std::endl;

  Solution sol1, sol2;
  sol1.resize(9);
  sol2.resize(9);

  for(int i = 0; i < sol1.size(); i++)
    sol1[i] = i;

  sol2[0] = 3;
  sol2[1] = 4;
  sol2[2] = 1;
  sol2[3] = 0;
  sol2[4] = 7;
  sol2[5] = 6;
  sol2[6] = 5;
  sol2[7] = 8;
  sol2[8] = 2;

  std::cout << sol1 << std::endl;
  std::cout << sol2 << std::endl;

  eoPartiallyMappedXover<Solution> xover;
  xover(sol1, sol2);

  std::cout << "apres" << std::endl;
  std::cout << sol1 << std::endl;
  std::cout << sol2 << std::endl;

  int verif[9];
  for(int i = 0; i < sol1.size(); i++)
    verif[i] = -1;

  for(int i = 0; i < sol1.size(); i++)
    verif[ sol1[i] ] = 1;

  for(int i = 0; i < sol1.size(); i++)
    assert(verif[i] != -1);


  for(int i = 0; i < sol2.size(); i++)
    verif[i] = -1;

  for(int i = 0; i < sol2.size(); i++)
    verif[ sol2[i] ] = 1;

  for(int i = 0; i < sol2.size(); i++)
    assert(verif[i] != -1);

  std::cout << "[t-eoPartiallyMappedXover] => OK" << std::endl;

  return EXIT_SUCCESS;
}
