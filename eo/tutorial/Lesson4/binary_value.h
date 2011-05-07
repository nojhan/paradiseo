#include <eo>

//-----------------------------------------------------------------------------

/** Just a simple function that takes binary value of a chromosome and sets
    the fitnes.
    @param _chrom A binary chromosome
*/

template <class Chrom> double binary_value(const Chrom& _chrom)
{
  double sum = 0;
  for (unsigned i = 0; i < _chrom.size(); i++)
    if (_chrom[i])
	sum += _chrom[i];
  return sum;
}

struct BinaryValue
{
  template <class Chrom> void operator()(Chrom& _chrom)
  {
    _chrom.fitness(binary_value(_chrom));
  }
};
