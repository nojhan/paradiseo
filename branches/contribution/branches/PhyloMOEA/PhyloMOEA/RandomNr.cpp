/*******************************************************************************
	RandomNr.cpp

		last change: 01/20/1999

		version: 0.0.0

		design:	Eckart Zitzler
			Paul E. Sevinc

		implementation:	Paul E. Sevinc

		(c) 1998-1999:	Computer Engineering and Networks Laboratory
				Swiss Federal Institute of Technology Zurich

		description:
			See RandomNr.h
*******************************************************************************/

#include "RandomNr.h"

#include <climits>
#include <cstddef>
#include <ctime>
#include <cmath>
#include <iostream>


using namespace std;

static long idum2=123456789;
static long iy=0;
static long iv[NTAB];
static long idum=0;


RandomNr::RandomNr()
{
	z = static_cast< long >( time( 0 ) ); // see <ctime>
        iy = idum = 0;
        sran1(z);
}


RandomNr::RandomNr( long seed )
	: z( seed )
{
        sran2(seed);
}


void
RandomNr::setSeed( long seed )
{
	z = seed;
        sran2(seed);
}


double
RandomNr::uniform01()
{
// see Reiser, Martin / Wirth, Niklaus.
//	- Programming in Oberon: Steps beyond Pascal and Modula.

	const int	a = 16807,
			m = LONG_MAX,	// see <climits> - replace by
					// numeric_limits< long >::max()
					// when <limits> is available
			q = m / a,
			r = m % a;

	long	gamma;

	gamma = a * ( z % q ) - r * ( z / q );
	z = ( gamma > 0 ) ? gamma : ( gamma + m );
	return z * ( 1.0 / m );

// NOTE: we're in trouble if at some point z becomes -LONG_MAX, 0, or LONG_MAX...
}


void
RandomNr::sran1(unsigned int seed) {
  int j;
  long k;

  idum = seed;
  if (idum == 0) idum=1;
  if (idum < 0) idum = -idum;
  for (j=NTAB+7;j>=0;j--) {
    k=(idum)/IQ;
    idum=IA*(idum-k*IQ)-IR*k;
    if (idum < 0) idum += IM;
    if (j < NTAB) iv[j] = idum;
  }
  iy=iv[0];
}

double
RandomNr::ran1() {
  int j;
  long k;
  double temp;
  k=(idum)/IQ;
  idum=IA*(idum-k*IQ)-IR*k;
  if (idum < 0) idum += IM;
  j=iy/NDIV;
  iy=iv[j];
  iv[j] = idum;
  if ((temp=AM*iy) > RNMX) return RNMX;
  else return temp;
}

void
RandomNr::sran2(unsigned int seed) {
  int j;
  long k;
  idum = static_cast<long>(seed);
  if (idum == 0) idum=1;
  if (idum < 0) idum = -idum;
  idum2=(idum);
  for (j=NTAB+7;j>=0;j--) {
    k=(idum)/IQ1;
    idum=IA1*(idum-k*IQ1)-k*IR1;
    if (idum < 0) idum += IM1;
    if (j < NTAB) iv[j] = idum;
  }
  iy=iv[0];
}

double
RandomNr::ran2() {
  int j;
  long k;
  float temp;
  k=(idum)/IQ1;
  idum=IA1*(idum-k*IQ1)-k*IR1;
  if (idum < 0) idum += IM1;
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;
  if (idum2 < 0) idum2 += IM2;
  j=iy/NDIV;
  iy=iv[j]-idum2;
  iv[j] = idum;
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) {
     return RNMX;
  }
  else{
    return temp;
  }
}



bool
RandomNr::flipP( double p )
{
	return ran2() < p ? true : false;
}


unsigned int
RandomNr::uniform0Max( unsigned int max )	// exclusive
{
      	return static_cast< unsigned short >( max * ran2() );
}


int
RandomNr::uniformMinMax(	int	min,	// inclusive
				int	max )	// exclusive
{
	return min >= 0
       		? static_cast< int >( min + ( max - min ) * ran2() )
		: static_cast< int >( min - 1 + ( max - min ) * ran2() );

}

int
RandomNr::gaussMinMax(	int	min,	// inclusive
				int	max )	// exclusive
{
        double tmp = UnitGaussian();
        if ((max*tmp > max) || ( max*tmp < min))
        {
           return max;
        }
	return min >= 0
       		? static_cast< int >( max  * tmp )
		: static_cast< int >( max  * tmp );

}




double
RandomNr::doubleuniformMinMax(	double	min,	// inclusive
				double	max )	// exclusive
{
	return static_cast< double >( min + ( max - min ) * ran2() );

}

double RandomNr::UnitGaussian(){
  static int cached=0;
  static double cachevalue;
  if(cached == 1){
    cached = 0;
    return cachevalue;
  }

  double rsquare, factor, var1, var2;
  do{
    var1 = 2.0 * ran2() - 1.0;
    var2 = 2.0 * ran2() - 1.0;

    rsquare = var1*var1 + var2*var2;
  } while(rsquare >= 1.0 || rsquare == 0.0);

  double val = -2.0 * log(rsquare) / rsquare;
  if(val > 0.0) factor = sqrt(val);
  else           factor = 0.0;	// should not happen, but might due to roundoff

  cachevalue = var1 * factor;
  cached = 1;

  return (var2 * factor);
}

void RandomNr::resumo(){
  cout << "semilla    =" << z << endl;
}



