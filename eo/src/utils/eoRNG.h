/** Random number generator adapted from (see comments below)

The random number generator is modified into a class
by Maarten Keijzer (mak@dhi.dk). Also added the Box-Muller
transformation to generate normal deviates.

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

Contact: eodev-main@lists.sourceforge.net
Old contact information: todos@geneura.ugr.es, http://geneura.ugr.es
*/

#ifndef EO_RANDOM_NUMBER_GENERATOR
#define EO_RANDOM_NUMBER_GENERATOR

/** @addtogroup Random
 * @{
 * */

# if (defined _MSC_VER)
/** uint32_t is an unsigned integer type capable of holding 32 bits.

 In the applicatione here exactly 32 are used.
 64 bits might be better on an Alpha or other 64 bit systems with GCC at high
 optimization levels so feel free to try your options and see what's best for
 you.
*/
typedef unsigned long uint32_t;
#else
#if (! defined __sun)
// The C99-standard defines uint32_t to be declared in stdint.h, but some
// systems don't have that and implement it in inttypes.h.
#include <stdint.h>
#else
#include <inttypes.h>
#endif
#endif

#include <cmath>
#include <vector>
#include "eoPersistent.h"
#include "eoObject.h"


/** Random Number Generator

@class eoRng eoRNG.h utils/eoRNG.h

eoRng is a persistent class that uses the ``Mersenne Twister'' random
number generator MT19937 for generating random numbers. The various
member functions implement useful functions for evolutionary
algorithms. Included are: rand(), random(), flip() and normal().

EO provides a global random number generator <tt>rng</tt> that is seeded by the
current UNIX time at program start. Moreover some global convenience functions
are provided that use the global random number generator: <tt>random</tt>,
<tt>normal</tt>.

@warning If you want to repeatedly generated the same sequence of pseudo-random
numbers, you should always reseed the generator at the beginning of your code.



<h1>Documentation in original file</h1>

This is the ``Mersenne Twister'' random number generator MT19937, which
generates pseudorandom integers uniformly distributed in 0..(2^32 - 1) starting
from any odd seed in 0..(2^32 - 1). This version is a recode by Shawn Cokus
(Cokus@math.washington.edu) on March 8, 1998 of a version by Takuji Nishimura
(who had suggestions from Topher Cooper and Marc Rieffel in July-August 1997).

Effectiveness of the recoding (on Goedel2.math.washington.edu, a DEC Alpha
running OSF/1) using GCC -O3 as a compiler: before recoding: 51.6 sec. to
generate 300 million random numbers; after recoding: 24.0 sec. for the same
(i.e., 46.5% of original time), so speed is now about 12.5 million random number
generations per second on this machine.

According to the URL <http://www.math.keio.ac.jp/~matumoto/emt.html> (and
paraphrasing a bit in places), the Mersenne Twister is ``designed with
consideration of the flaws of various existing generators,'' has a period of
2^19937 - 1, gives a sequence that is 623-dimensionally equidistributed, and
``has passed many std::stringent tests, including the die-hard test of G.
Marsaglia and the load test of P. Hellekalek and S. Wegenkittl.'' It is
efficient in memory usage (typically using 2506 to 5012 bytes of static data,
depending on data type sizes, and the code is quite short as well). It generates
random numbers in batches of 624 at a time, so the caching and pipelining of
modern systems is exploited. It is also divide- and mod-free.

The code as Shawn received it included the following notice: <tt>Copyright (C)
1997 Makoto Matsumoto and Takuji Nishimura. When you use this, send an e-mail to
<matumoto@math.keio.ac.jp> with an appropriate reference to your work.</tt> It
would be nice to Cc: <Cokus@math.washington.edu> and
<eodev-main@lists.sourceforge.net> when you write.


<h1>Portability</h1>

Note for people porting EO to other platforms: please make sure that the type
uint32_t in the file eoRNG.h is exactly 32 bits long. It may in principle be
longer, but not shorter. If it is longer, file compatibility between EO on
different platforms may be broken.
*/
class eoRng : public eoObject, public eoPersistent
{
public :

    /** Constructor

    @param s Random seed; if you want another seed, use reseed.

    @see reseed for details on usage of the seeding value.
    */
    eoRng(uint32_t s)
        : state(0), next(0), left(-1), cached(false)
        {
            state = new uint32_t[N+1];
            initialize(2*s);
        }

    /** Destructor */
    ~eoRng()
        {
            delete [] state;
        }

    /** Re-initializes the Random Number Generator.

    WARNING: Jeroen Eggermont <jeggermo@liacs.nl> noticed that initialize does
    not differentiate between odd and even numbers, therefore the argument to
    reseed is now doubled before being passed on.

    Manually divide the seed by 2 if you want to re-run old runs

    @version MS. 5 Oct. 2001
    */
    void reseed(uint32_t s)
        {
            initialize(2*s);
        }

    /* FIXME remove in next release
    ** Re-initializes the Random Number Generator

    This is the traditional seeding procedure. This version is deprecated and
    only provided for compatibility with old code. In new projects you should
    use reseed.

    @see reseed for details on usage of the seeding value.

    @version old version (deprecated)
    *
    void oldReseed(uint32_t s)
        {
            initialize(s);
        }
    */

    /** Random number from unifom distribution

    @param m Define interval for random number to [0, m)
    @return random number in the range [0, m)
    */
    double uniform(double m = 1.0)
        { // random number between [0, m]
            return m * double(rand()) / double(1.0 + rand_max());
        }

    /** Random number from unifom distribution

    @param min Define minimum for interval in the range [min, max)
    @param max Define maximum for interval in the range [min, max)
    @return random number in the range [min, max)
    */
    double uniform(double min, double max)
        { // random number between [min, max]
            return min + uniform(max - min);
        }

    /** Random integer number from unifom distribution

    @param m Define interval for random number to [0, m)
    @return random integer in the range [0, m)
    */
    uint32_t random(uint32_t m)
        {
            // C++ Standard (4.9 Floatingintegral conversions [conv.fpint])
            // defines floating point to integer conversion as truncation
            // ("rounding towards zero"): "An rvalue of a floating point type
            // can be converted to an rvalue of an integer type. The conversion
            // truncates; that is, the fractional part is discarded"
            return uint32_t(uniform() * double(m));
        }

    /** Biased coin toss

    This tosses a biased coin such that flip(x/100.0) will true x% of the time

    @param bias The coins' bias (the \e x above)
    @return The result of the biased coin toss
    */
    bool flip(double bias=0.5)
        {
            return uniform() < bias;
        }

    /** Gaussian deviate

    Zero mean Gaussian deviate with standard deviation 1.
    Note: Use the Marsaglia polar method.

    @return Random Gaussian deviate
    */
    double normal();

    /** Gaussian deviate

    Gaussian deviate with zero mean and specified standard deviation.

    @param stdev Standard deviation for Gaussian distribution
    @return Random Gaussian deviate
    */
    double normal(double stdev)
        { return stdev * normal(); }

    /** Gaussian deviate

    Gaussian deviate with specified mean and standard deviation.

    @param mean Mean for Gaussian distribution
    @param stdev Standard deviation for Gaussian distribution
    @return Random Gaussian deviate
    */
    double normal(double mean, double stdev)
        { return mean + normal(stdev); }

    /** Random numbers using a negative exponential distribution

    @param mean Mean value of distribution
    @return Random number from a negative exponential distribution
    */
    double negexp(double mean)
        {
            return -mean*log(double(rand()) / rand_max());
        }

    /**
    rand() returns a random number in the range [0, rand_max)
    */
    uint32_t rand();

    /**
    rand_max() the maximum returned by rand()
    */
    uint32_t rand_max() const { return uint32_t(0xffffffff); }

    /** Roulette wheel selection

    roulette_wheel(vec, total = 0) does a roulette wheel selection
    on the input std::vector vec. If the total is not supplied, it is
    calculated. It returns an integer denoting the selected argument.
    */
    template <typename TYPE>
    int roulette_wheel(const std::vector<TYPE>& vec, TYPE total = 0)
        {
            if (total == 0)
            { // count
                for (unsigned    i = 0; i < vec.size(); ++i)
                    total += vec[i];
            }
            double fortune = uniform() * total;
            int i = 0;
            while (fortune >= 0)
            {
                fortune -= vec[i++];
            }
            return --i;
        }


    /** Randomly select element from vector.

    @return Uniformly chosen element from the vector.
    */
    template <typename TYPE>
    const TYPE& choice(const std::vector<TYPE>& vec)
        { return vec[random(vec.size())]; }


    /** Randomly select element from vector.

    @overload

    Provide a version returning a non-const element reference.

    @return Uniformly chosen element from the vector.

    @warning Changing the return value does alter the vector.
    */
    template <typename TYPE>
    TYPE& choice(std::vector<TYPE>& vec)
        { return vec[random(vec.size())]; }

    /** @brief Print RNG

    @param _os Stream to print RNG on
    */
    void printOn(std::ostream& _os) const
        {
            for (int i = 0; i < N; ++i)
            {
                _os << state[i] << ' ';
            }
            _os << int(next - state) << ' ';
            _os << left << ' ' << cached << ' ' << cacheValue;
        }

    /** @brief Read RNG

    @param _is Stream to read RNG from
    */
    void readFrom(std::istream& _is)
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

    std::string className() const { return "Mersenne-Twister"; }

private:

    uint32_t restart();

    /* @brief Initialize state

    We initialize state[0..(N-1)] via the generator

    <tt>x_new = (69069 * x_old) mod 2^32</tt>

    from Line 15 of Table 1, p. 106, Sec. 3.3.4 of Knuth's _The Art of Computer
    Programming_, Volume 2, 3rd ed.

    Notes (SJC): I do not know what the initial state requirements of the
    Mersenne Twister are, but it seems this seeding generator could be better.
    It achieves the maximum period for its modulus (2^30) iff x_initial is odd
    (p. 20-21, Sec. 3.2.1.2, Knuth); if x_initial can be even, you have
    sequences like 0, 0, 0, ...; 2^31, 2^31, 2^31, ...; 2^30, 2^30, 2^30, ...;
    2^29, 2^29 + 2^31, 2^29, 2^29 + 2^31, ..., etc. so I force seed to be odd
    below.

    Even if x_initial is odd, if x_initial is 1 mod 4 then

    the          lowest bit of x is always 1,
    the  next-to-lowest bit of x is always 0,
    the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    the 3rd-from-lowest bit of x 4-cycles        ... 0 1 1 0 0 1 1 0 ... ,
    the 4th-from-lowest bit of x has the 8-cycle ... 0 0 0 1 1 1 1 0 ... ,
    ...

    and if x_initial is 3 mod 4 then

    the          lowest bit of x is always 1,
    the  next-to-lowest bit of x is always 1,
    the 2nd-from-lowest bit of x alternates      ... 0 1 0 1 0 1 0 1 ... ,
    the 3rd-from-lowest bit of x 4-cycles        ... 0 0 1 1 0 0 1 1 ... ,
    the 4th-from-lowest bit of x has the 8-cycle ... 0 0 1 1 1 1 0 0 ... ,
    ...

    The generator's potency (min. s>=0 with (69069-1)^s = 0 mod 2^32) is 16,
    which seems to be alright by p. 25, Sec. 3.2.1.3 of Knuth. It also does well
    in the dimension 2..5 spectral tests, but it could be better in dimension 6
    (Line 15, Table 1, p. 106, Sec. 3.3.4, Knuth).

    Note that the random number user does not see the values generated here
    directly since restart() will always munge them first, so maybe none of all
    of this matters. In fact, the seed values made here could even be
    extra-special desirable if the Mersenne Twister theory says so-- that's why
    the only change I made is to restrict to odd seeds.
    */
    void initialize(uint32_t seed);

    /** @brief Array for the state */
    uint32_t *state;

    /** Pointer to next available random number */
    uint32_t *next;

    /** Number of random numbers currently left */
    int left;

    /** @brief Is there a valid cached value for the normal distribution? */
    bool cached;

    /** @brief Cached value for normal distribution? */
    double cacheValue;

    /** @brief Size of the state-vector */
    static const int N;

    /** Internal constant */
    static const int M;

    /** @brief Magic constant */
    static const uint32_t K;


    /** @brief Copy constructor

    Private copy ctor and assignment operator to make sure that nobody
    accidentally copies the random number generator. If you want similar RNG's,
    make two RNG's and initialize them with the same seed.

    As it cannot be called, we do not provide an implementation.
    */
    eoRng(const eoRng&);

    /** @brief Assignment operator

    @see Copy constructor eoRng(const eoRng&).
    */
    eoRng& operator=(const eoRng&);
};
/** @example t-eoRNG.cpp
 */



namespace eo
{
    /** The one and only global eoRng object */
    extern eoRng rng;
}
using eo::rng;

/** @} */




// Implementation of some eoRng members.... Don't mind the mess, it does work.

#define hiBit(u)       ((u) & 0x80000000U)   // mask all but highest   bit of u
#define loBit(u)       ((u) & 0x00000001U)   // mask all but lowest    bit of u
#define loBits(u)      ((u) & 0x7FFFFFFFU)   // mask     the highest   bit of u
#define mixBits(u, v)  (hiBit(u)|loBits(v))  // move hi bit of u to hi bit of v


inline void eoRng::initialize(uint32_t seed)
{
    left = -1;

    register uint32_t x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
    register int j;

    for(left=0, *s++=x, j=N; --j;
        *s++ = (x*=69069U) & 0xFFFFFFFFU) ;
}



inline uint32_t eoRng::restart()
{
    register uint32_t *p0=state, *p2=state+2, *pM=state+M, s0, s1;
    register int j;

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



inline uint32_t eoRng::rand()
{
    if(--left < 0)
        return(restart());
    uint32_t y  = *next++;
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    return(y ^ (y >> 18));
}



inline double eoRng::normal()
{
    if (cached) {
        cached = false;
        return cacheValue;
    }
    double rSquare, var1, var2;
    do {
        var1 = 2.0 * uniform() - 1.0;
        var2 = 2.0 * uniform() - 1.0;
        rSquare = var1 * var1 + var2 * var2;
    } while (rSquare >= 1.0 || rSquare == 0.0);
    double factor = sqrt(-2.0 * log(rSquare) / rSquare);
    cacheValue = var1 * factor;
    cached = true;
    return (var2 * factor);
}



namespace eo
{
    /** @brief Random function

    This is a convenience function for generating random numbers using the
    global eo::rng object.

    Templatized random function, returns a random double in the range [min, max).
    It works with most basic types such as:
    - char
    - int (short, long, signed and unsigned)
    - float, double

    @param min Minimum for distribution
    @param max Maximum for distribution

    @see random(const T& max)
    */
    template <typename T>
    inline T random(const T& min, const T& max) {
        return static_cast<T>(rng.uniform() * (max-min)) + min; }

    /** @brief Random function

    @overload

    This is a convenience function for generating random numbers using the
    global eo::rng object.

    Templatized random function, returns a random double in the range [0, max).
    It works with most basic types such as:
    - char
    - int (short, long, signed and unsigned)
    - float, double

    @param max Maximum for distribution

    @see random(const T& min, const T& max)
    */
    template <typename T>
    inline T random(const T& max) {
        return static_cast<T>(rng.uniform() * max); }

    /** Normal distribution

    This is a convenience function for generating random numbers using the
    global eo::rng object.

    @return ormally distributed random number
    */
    inline double normal() { return rng.normal(); }
}


#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
