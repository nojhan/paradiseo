#include <vector>
//-----------------------------------------------------------------------------
/** Just a simple function that takes an vector<double> and sets the fitnes 
    to the sphere function. Please use doubles not float!!!
    @param _ind A floatingpoint vector 
*/

// INIT
double real_value(const std::vector<double>& _ind)
{
  double sum = 0;
  for (unsigned i = 0; i < _ind.size(); i++)
      {
	  if ( (_ind[i]<0) || (_ind[i]>1) )
	      cout << "Sorti des bornes: " << _ind[i] << " ";
	  sum += _ind[i] * _ind[i];
      }
  return -sum;
}



