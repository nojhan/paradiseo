/*
*	Random number generator adapted from (see comments below)
*   
*   The random number generator is modified into a class
*   by Maarten Keijzer (mak@dhi.dk). Also added the Box-Muller
*   transformation to generate normal deviates.
*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
*/

/* ************ DOCUMENTATION IN ORIGINAL FILE *********************/

// This is the ``Mersenne Twister'' random number generator MT19937, which
// generates pseudorandom integers uniformly distributed in 0..(2^32 - 1)
// starting from any odd seed in 0..(2^32 - 1).  This version is a recode
// by Shawn Cokus (Cokus@math.washington.edu) on March 8, 1998 of a version by
// Takuji Nishimura (who had suggestions from Topher Cooper and Marc Rieffel in
// July-August 1997).
//
// Effectiveness of the recoding (on Goedel2.math.washington.edu, a DEC Alpha
// running OSF/1) using GCC -O3 as a compiler: before recoding: 51.6 sec. to
// generate 300 million random numbers; after recoding: 24.0 sec. for the same
// (i.e., 46.5% of original time), so speed is now about 12.5 million random
// number generations per second on this machine.
//
// According to the URL <http://www.math.keio.ac.jp/~matumoto/emt.html>
// (and paraphrasing a bit in places), the Mersenne Twister is ``designed
// with consideration of the flaws of various existing generators,'' has
// a period of 2^19937 - 1, gives a sequence that is 623-dimensionally
// equidistributed, and ``has passed many stringent tests, including the
// die-hard test of G. Marsaglia and the load test of P. Hellekalek and
// S. Wegenkittl.''  It is efficient in memory usage (typically using 2506
// to 5012 bytes of static data, depending on data type sizes, and the code
// is quite short as well).  It generates random numbers in batches of 624
// at a time, so the caching and pipelining of modern systems is exploited.
// It is also divide- and mod-free.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Library General Public License as published by
// the Free Software Foundation (either version 2 of the License or, at your
// option, any later version).  This library is distributed in the hope that
// it will be useful, but WITHOUT ANY WARRANTY, without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
// the GNU Library General Public License for more details.  You should have
// received a copy of the GNU Library General Public License along with this
// library; if not, write to the Free Software Foundation, Inc., 59 Temple
// Place, Suite 330, Boston, MA 02111-1307, USA.
//
// The code as Shawn received it included the following notice:
//
//   Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.  When
//   you use this, send an e-mail to <matumoto@math.keio.ac.jp> with
//   an appropriate reference to your work.
//
// It would be nice to CC: <Cokus@math.washington.edu> when you write.
//

//
// uint32 must be an unsigned integer type capable of holding at least 32
// bits; exactly 32 should be fastest, but 64 is better on an Alpha with
// GCC at -O3 optimization so try your options and see what's best for you
//

/* ************ END DOCUMENTATION IN ORIGINAL FILE *********************/


#ifndef EO_RANDOM_NUMBER_GENERATOR
#define EO_RANDOM_NUMBER_GENERATOR

#include <ctime>

#include <eoPersistent.h>
#include <eoObject.h>

// TODO: check for various compilers if this is exactly 32 bits
// Unfortunately MSVC's preprocessor does not comprehends sizeof()
// so neat preprocessing tricks will not work

typedef unsigned long uint32; // Compiler and platform dependent!

//-----------------------------------------------------------------------------
// eoRng
//-----------------------------------------------------------------------------
/**
eoRng is a persitent class that uses the ``Mersenne Twister'' random number generator MT19937
for generating random numbers. The various member functions implement useful functions
for evolutionary algorithms. Included are: rand(), random(), flip() and normal().

Note for people porting EO to other platforms: please make sure that the typedef
uint32 in the file eoRng.h is exactly 32 bits long. It may be longer, but not
shorter. If it is longer, file compatibility between EO on different platforms
may be broken.
*/
class eoRng  : public eoObject, public eoPersistent
{
public :
  /**
     ctor takes a random seed; if you want another seed, use reseed
     @see reseed
  */

  eoRng(uint32 s = (uint32) time(0) ) : state(0), next(0), left(-1), cached(false), N(624), M(397), K(0x9908B0DFU)  {
    state = new uint32[N+1];
    initialize(s);
  } 
 
 ~eoRng(void)
   {
     delete [] state;
   }
 
 /**
    Re-initializes the Random Number Generator.
 */
 void reseed(uint32 s) 
   { 
     initialize(s); 
   }
 
 /**
    uniform(m = 1.0) returns a random double in the range [0, m)
 */
 double uniform(double m = 1.0)
   { // random number between [0, m]
     return m * double(rand()) / double(rand_max());
   }
 
 /**
    random() returns a random integer in the range [0, m)
 */
 uint32 random(uint32 m)
   {
     return uint32(uniform() * double(m));
   }
 
 /**
    flip() tosses a biased coin such that flip(x/100.0) will 
    returns true x% of the time
 */
 bool flip(float bias)
   {
     return uniform() < bias;
   }
 
 /**
    normal() zero mean gaussian deviate with standard deviation of 1
 */
 double normal(void);        // gaussian mutation, stdev 1       
 
 /**
    normal(stdev) zero mean gaussian deviate with user defined standard deviation
 */
 double normal(double stdev)  
   {
     return stdev * normal();
   }
 
 /**
    normal(mean, stdev) user defined mean gaussian deviate with user defined standard deviation
 */
 double normal(double mean, double stdev)
   {
     return mean + normal(stdev);
   }
 
 /**
    rand() returns a random number in the range [0, rand_max)
 */
 uint32 rand(); 
 
 /**
    rand_max() the maximum returned by rand()
 */
 uint32 rand_max(void) const { return (uint32) 0xffffffff; }		
 
 /**
    roulette_wheel(vec, total = 0) does a roulette wheel selection 
    on the input vector vec. If the total is not supplied, it is
    calculated. It returns an integer denoting the selected argument.
 */
 template <class T>
   int roulette_wheel(const std::vector<T>& vec, T total = 0)
   {
     if (total == 0)
       { // count 
	 for (unsigned i = 0; i < vec.size(); ++i)
	   total += vec[i];
       }
     
     float change = uniform() * total;
     
     int i = 0;
     
     while (change > 0)
       {
	 change -= vec[i++];
       }
     
     return --i;
   }

 ///
 void printOn(ostream& _os) const
   {
     for (int i = 0; i < N; ++i)
       {
	 _os << state[i] << ' ';
       }
     _os << int(next - state) << ' ';
     _os << left << ' ' << cached << ' ' << cacheValue;
   }
 
 ///
 void readFrom(istream& _is)
   {
     for (int i = 0; i < N; ++i)
       {
	 _is >> state[i];
       }
     
     int n;
     _is >> n;
     next = state + n;
     
     _is >> left;
     _is >> cached;
     _is >> cacheValue;
   }


private :
  uint32 restart(void);
 void initialize(uint32 seed);
 
 uint32* state; // the array for the state 
 uint32* next;
 int left;				  
 
 bool cached;
 float cacheValue;
 
 const int N;
 const int M;
 const uint32 K; // a magic constant

 /**
	Private copy ctor and assignment operator to make sure that
	nobody accidentally copies the random number generator.
	If you want similar RNG's, make two RNG's and initialize
	them with the same seed.
 */
  eoRng (const eoRng&);				// no implementation
  eoRng& operator=(const eoRng&);	// dito
};

/**
	The one and only global eoRng object
*/
static eoRng rng;

/**
   The class uniform_generator can be used in the STL generate function
   to easily generate random floats and doubles between [0, _max). _max
   defaults to 1.0
*/
template <class T = double> class uniform_generator
{
  public :
    uniform_generator(T _max = T(1.0), eoRng& _rng = rng) : maxim(_max), uniform(_rng) {}
  
  virtual T operator()(void) { return (T) uniform.uniform(maxim); } 
  private :
    T maxim;
  eoRng& uniform;
};

/**
   The class random_generator can be used in the STL generate function
   to easily generate random ints between [0, _max).
*/
template <class T = uint32> class random_generator
{
  public :
    random_generator(int _max, eoRng& _rng = rng) : maxim(_max), random(_rng) {}
  
  virtual T operator()(void) { return (T) random.random(max); }
  
  private :
    T maxim;
  eoRng& random;
};

/**
   The class normal_generator can be used in the STL generate function
   to easily generate gaussian distributed floats and doubles. The user
   can supply a standard deviation which defaults to 1.
*/
template <class T = double> class normal_generator
{
  public :
    normal_generator(T _stdev = T(1.0), eoRng& _rng = rng) : stdev(_stdev), normal(_rng) {}
  
  virtual T operator()(void) { return (T) normal.normal(stdev); }
  
  private :
    T stdev;
  eoRng& normal;
};

// Implementation of some eoRng members.... Don't mind the mess, it does work.


#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v

inline void eoRng::initialize(uint32 seed)
 {
    //
    // We initialize state[0..(N-1)] via the generator
    //
    //   x_new = (69069 * x_old) mod 2^32
    //
    // from Line 15 of Table 1, p. 106, Sec. 3.3.4 of Knuth's
    // _The Art of Computer Programming_, Volume 2, 3rd ed.
    //
    // Notes (SJC): I do not know what the initial state requirements
    // of the Mersenne Twister are, but it seems this seeding generator
    // could be better.  It achieves the maximum period for its modulus
    // (2^30) iff x_initial is odd (p. 20-21, Sec. 3.2.1.2, Knuth); if
    // x_initial can be even, you have sequences like 0, 0, 0, ...;
    // 2^31, 2^31, 2^31, ...; 2^30, 2^30, 2^30, ...; 2^29, 2^29 + 2^31,
    // 2^29, 2^29 + 2^31, ..., etc. so I force seed to be odd below.
    //
    // Even if x_initial is odd, if x_initial is 1 mod 4 then
    //
    //   the          lowest bit of x is always 1,
    //   the  next-to-lowest bit of x is always 0,
    //   the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    //   the 3rd-from-lowest bit of x 4-cycles        ... 0 1 1 0 0 1 1 0 ... ,
    //   the 4th-from-lowest bit of x has the 8-cycle ... 0 0 0 1 1 1 1 0 ... ,
    //    ...
    //
    // and if x_initial is 3 mod 4 then
    //
    //   the          lowest bit of x is always 1,
    //   the  next-to-lowest bit of x is always 1,
    //   the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    //   the 3rd-from-lowest bit of x 4-cycles        ... 0 0 1 1 0 0 1 1 ... ,
    //   the 4th-from-lowest bit of x has the 8-cycle ... 0 0 1 1 1 1 0 0 ... ,
    //    ...
    //
    // The generator's potency (min. s>=0 with (69069-1)^s = 0 mod 2^32) is
    // 16, which seems to be alright by p. 25, Sec. 3.2.1.3 of Knuth.  It
    // also does well in the dimension 2..5 spectral tests, but it could be
    // better in dimension 6 (Line 15, Table 1, p. 106, Sec. 3.3.4, Knuth).
    //
    // Note that the random number user does not see the values generated
    // here directly since restart() will always munge them first, so maybe
    // none of all of this matters.  In fact, the seed values made here could
    // even be extra-special desirable if the Mersenne Twister theory says
    // so-- that's why the only change I made is to restrict to odd seeds.
    //

   left = -1;
   
   register uint32 x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
   register int    j;
   
   for(left=0, *s++=x, j=N; --j;
       *s++ = (x*=69069U) & 0xFFFFFFFFU);
 }


inline uint32 eoRng::restart(void)
{
  register uint32 *p0=state, *p2=state+2, *pM=state+M, s0, s1;
  register int    j;
  
  left=N-1, next=state+1;
  
  for(s0=state[0], s1=state[1], j=N-M+1; --j; s0=s1, s1=*p2++)
    *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
  
  for(pM=state, j=M; --j; s0=s1, s1=*p2++)
    *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
  
  s1=state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
  s1 ^= (s1 >> 11);
  s1 ^= (s1 <<  7) & 0x9D2C5680U;
  s1 ^= (s1 << 15) & 0xEFC60000U;
  return(s1 ^ (s1 >> 18));
}


inline uint32 eoRng::rand(void)
 {
   uint32 y;
   
   if(--left < 0)
     return(restart());
   
   y  = *next++;
   y ^= (y >> 11);
   y ^= (y <<  7) & 0x9D2C5680U;
   y ^= (y << 15) & 0xEFC60000U;
   return(y ^ (y >> 18));
 }

inline double eoRng::normal(void)
{	
  if (cached)
    {
      cached = false;
      return cacheValue;
    }
  
  float rSquare, factor, var1, var2;
  
  do
	{
	  var1 = 2.0 * uniform() - 1.0;
	  var2 = 2.0 * uniform() - 1.0;
	  
	  rSquare = var1 * var1 + var2 * var2;
	} 
  while (rSquare >= 1.0 || rSquare == 0.0);
  
  factor = sqrt(-2.0 * log(rSquare) / rSquare);
  
  cacheValue = var1 * factor;
  cached = true;
  
  return (var2 * factor);
}

#endif
