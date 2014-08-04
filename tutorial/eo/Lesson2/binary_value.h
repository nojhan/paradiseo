#include <paradiseo/eo.h>

//-----------------------------------------------------------------------------

/** Just a simple function that takes binary value of a chromosome and sets
    the fitnes.
    @param _chrom A binary chromosome
*/
// INIT
double binary_value(const std::vector<bool>& _chrom)
{
  double sum = 0;
  for (unsigned i = 0; i < _chrom.size(); i++)
    sum += _chrom[i];
  return sum;
}
