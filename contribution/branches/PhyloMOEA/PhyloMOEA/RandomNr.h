/*******************************************************************************
	RandomNr.h
		
		last change: 01/20/1999
						
		version: 0.0.0
			
		design:	Eckart Zitzler
			Paul E. Sevinc
					
		implementation:	Paul E. Sevinc
			
		(c) 1998-1999:	Computer Engineering and Networks Laboratory
				Swiss Federal Institute of Technology Zurich
					
		description:
			RandomNr is a helper class for pseudo-random number
			generation that doesn't need to be subclassed unless
			one wants to replace the algorithm used or add
			another method.

			Usually, one only instance per optimization problem
			is necessary, and classes or functions that make use
			of random numbers keep a reference to that instance.
*******************************************************************************/

#ifndef RANDOM_NR_H
#define RANDOM_NR_H

#include <cstddef>
//#include "TIKEAFExceptions.h"


#define IM1 2147483563L
#define IM2 2147483399L
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014L
#define IA2 40692L
#define IQ1 53668L
#define IQ2 52774L
#define IR1 12211L
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

#define IA 16807L
#define IM 2147483647L
//#define AM (1.0/IM)
#define IQ 127773L
#define IR 2836L
//#define NTAB 32
//#define NDIV (1+(IM-1)/NTAB)
//#define EPS 1.2e-7
//#define RNMX (1.0-EPS)


//typedef unsigned int size_t;

/*static long iy = 0;
static long iv[NTAB];
static long idum = 0;*/


class RandomNr
{
	protected:
		long	z;

	public:
		// use the current time as seed
		RandomNr();

		// use the argument as seed
			RandomNr( long );

		void	setSeed( long );
		virtual ~RandomNr() {};
		// return the next uniformly distributed
		// random real in ]0; 1[
		virtual	double	uniform01();

			// return the next uniformly distributed
			// random integer in [Min; Max[
		int	uniformMinMax( int, int );
//					throw ( LimitsException );

		double  doubleuniformMinMax(double, double);

		int  gaussMinMax(int, int );	// exclusive


		double  UnitGaussian();

			// return the next uniformly distributed
			// random integer (size_t) in [0; Max[
		unsigned int uniform0Max( unsigned int );
//					throw ( LimitsException );

			// flip an unfair coin (head is up with
			// probability P), return true if head
			// is up and false otherwise
		bool	flipP( double );
			//throw ( ProbabilityException );

		void  sran1(unsigned int seed);
		double ran1();
		void  sran2(unsigned int seed);
		double ran2();
		void  resumo();
		long get_seed() { return z; }
};

#endif
