#include <eoESFullChrom.h>

//-----------------------------------------------------------------------------

typedef vector<double> Vec;

/** Just a simple function that takes an eoVector<float> and sets the fitnes 
    to -sphere (we'll see later how to minimize rather than maximize!)
    @param _ind A floatingpoint vector 
*/
float the_real_value(Vec& _ind)
{
  double sum = 0;      /* compute in double format, even if return a float */
  for (unsigned i = 0; i < _ind.size(); i++)
      sum += _ind[i] * _ind[i];
  return -sum;
}

typedef eoESFullChrom<float> Ind;

void real_value(Ind & _ind) {
    _ind.fitness( the_real_value(_ind) );
}
