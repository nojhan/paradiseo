#include <vector>
//-----------------------------------------------------------------------------


/** Just a simple function that takes an eoEsBase<double> and sets the fitnes
    to sphere
    @param _ind  vector<double>
*/

double real_value(const std::vector<double>& _ind)
{
  double sum = 0;
  for (unsigned i = 0; i < _ind.size(); i++)
      sum += _ind[i] * _ind[i];
  return sum/_ind.size();
}
