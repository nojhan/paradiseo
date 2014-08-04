//-----------------------------------------------------------------------------
// t-rng.cpp
//-----------------------------------------------------------------------------

// This file really needs to be implementes usign some stringent tests, for now
// we simply check that the impementation of some methods does generally work...

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <paradiseo/eo.h>
#include <paradiseo/eo/utils/eoRNG.h>


using namespace std;


int main()
{
    const size_t num(10000);
    double mean(100.);
    double sigma(5.);
    double sum(0.);
    for(size_t i=0; i<num; ++i)
	sum += abs(rng.normal(sigma));
    sum /= double(num);
    if(sum > sigma / 0.68) {
	cerr << "Normal distribution seems out of bounds; "
	     << "rerun to make sure it wasn't a statistical outlier" << endl;
	return -1;
    }
    sum = 0.;
    for(size_t i=0; i<num; ++i)
	sum += abs(rng.normal(mean, sigma) - mean);
    sum /= double(num);
    if(sum > sigma / 0.68) {
	cerr << "Normal distribution seems out of bounds; "
	     << "rerun to make sure it wasn't a statistical outlier" << endl;
	return -1;
    }
  return 0;
}


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
