#include <algorithm>

//-----------------------------------------------------------------------------

/** Just the simple function that takes binary value of a chromosome and sets
    the fitnes.
    @param _chrom A binary chromosome
*/

template <class Chrom> double binary_value(const Chrom& _chrom)
{
    double sum = 0.0;
    for (unsigned i=0; i<_chrom.size(); i++)
	sum += _chrom[i];
    return sum;
}
