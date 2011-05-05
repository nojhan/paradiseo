/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoSharing.h
   (c) Maarten Keijzer, Marc Schoenauer, 2001

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

    Contact: Marc.Schoenauer@inria.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoSharing_h
#define eoSharing_h

#include <eoPerf2Worth.h>
#include <utils/eoDistance.h>

/** Sharing is a perf2worth class that implements
 *  Goldberg and Richardson's basic sharing
*/

/** A helper class for Sharing - to  hold distances
 *
 * @ingroup Selectors
 * */
class dMatrix : public std::vector<double>
  {
  public:
    // Ctor : sets size
    dMatrix(unsigned _s) : std::vector<double>(_s*_s), rSize(_s) {}

    /** simple accessor */
    double operator()(unsigned _i, unsigned _j) const
    {
      return this->operator[](_i*rSize + _j);
    }

    /** reference - to set values */
    double & operator()(unsigned _i, unsigned _j)
    {
      return this->operator[](_i*rSize + _j);
    }

    /** just in case */
    void printOn(std::ostream & _os)
    {
      unsigned index=0;
      for (unsigned i=0; i<rSize; i++)
        {
          for (unsigned j=0; j<rSize; j++)
            _os << this->operator[](index++) << " " ;
          _os << std::endl;
        }
      _os << std::endl;
    }

    private:
      unsigned rSize;              // row size (== number of columns!)
  };


/** Sharing is a perf2worth class that implements
 *  Goldberg and Richardson's basic sharing
 *  see eoSharingSelect for how to use it
 * and test/t-eoSharing.cpp for a sample use of both
 * @ingroup Selectors
*/
template <class EOT>
class eoSharing : public eoPerf2Worth<EOT>
{
public:

    using eoPerf2Worth<EOT>::value;


  /* Ctor requires a distance - cannot have a default distance! */
  eoSharing(double _nicheSize, eoDistance<EOT> & _dist) : eoPerf2Worth<EOT>("Sharing"),
                                             nicheSize(_nicheSize),
                                             dist(_dist)
  {}

  /** Computes shared fitnesses
  */
    void operator()(const eoPop<EOT>& _pop)
    {
      unsigned i, j,
        pSize=_pop.size();
      if (pSize <= 1)
        throw std::runtime_error("Apptempt to do sharing with population of size 1");
      value().resize(pSize);
      std::vector<double> sim(pSize);      // to hold the similarities
      dMatrix distMatrix(pSize); // to hold the distances

      // compute the similarities (wrong name for distMatrix, I know)
      distMatrix(0,0)=1;
      for (i=1; i<pSize; i++)
        {
          distMatrix(i,i)=1;
          for (j=0; j<i; j++)
            {
              double d =  dist(_pop[i], _pop[j]);
              distMatrix(i,j) =
                distMatrix(j,i) = ( d>nicheSize ? 0 : 1-(d/nicheSize) );
            }
        }

      for (i=0; i<pSize; i++)
        {
          double sum=0.0;
          for (j=0; j<pSize; j++)
            sum += distMatrix(i,j);
          sim[i] = sum;
        }

      // now set the worthes values
      for (i = 0; i < _pop.size(); ++i)
        value()[i]=_pop[i].fitness()/sim[i];
    }
    // private data of class eoSharing
private:
  double nicheSize;
  eoDistance<EOT> & dist;            // specific distance
};



#endif
