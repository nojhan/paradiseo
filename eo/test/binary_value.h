#include <eo>

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

/** Just a simple function that takes binary value of a chromosome and sets
    the fitnes.
    @param _chrom A binary chromosome 
*/
void binary_value(Chrom& _chrom)
{
  float sum = 0;
  for (unsigned i = 0; i < _chrom.size(); i++)
    if (_chrom[i])
      sum += pow(2, _chrom.size() - i - 1);
  _chrom.fitness(sum);
}
