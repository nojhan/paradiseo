#include <es/eoEsBase.h>
//-----------------------------------------------------------------------------


/** Just a simple function that takes an eoEsBase<float> and sets the fitnes 
    to sphere 
    @param _ind A floatingpoint vector 
*/

double real_value(const eoEsBase<double>& _ind)
{
  double sum = 0;      /* compute in double format, even if return a float */
  for (unsigned i = 0; i < _ind.size(); i++)
      sum += _ind[i] * _ind[i];
  return sum;
}



