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
#define eoPerf2Worth_h

#include <eoPerf2Worth.h>
#include <utils/eoDistance.h>

/** Sharing is a perf2worth class that implements 
 *  Goldberg and Richardson's basic sharing
*/
// template <class EOT, class Dist = eoQuadDistance<EOT> >
template <class EOT>
class eoSharing : public eoPerf2Worth<EOT, double>
{
  public:
  /** Ctor with only nicheSize: will use the default eoQuadDistance */
  eoSharing(double _nicheSize) : eoPerf2Worth("Sharing"), 
				 nicheSize(_nicheSize), 
				 dist(repDist)
  {}

  eoSharing(double _nicheSize, Dist _dist) : eoPerf2Worth("Sharing"), 
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
	throw runtime_error("Apptempt to do sharing with population of size 1");
      value.resize(pSize);
      vector<double> sim(pSize);	   // to hold the similarities
      vector<double> distMatrix(pSize*(pSize-1)/2); // to hold the distances

      // compute the distances
      distMatrix(0,0)=0;
      for (i=1; i<pSize; i++)
	{
	  distMatrix(i,i)=0;
	  for (j=0; j<i; j++)
	    {
	      distMatrix(i,j) = distMatrix(j,i) = dist(_pop[i], _pop[j]);
	    }
	}

      // compute the similarities
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

  // helper class to  hold distances
  class dMatrix : public vector<double>
  {
  public:
    // Ctor : sets size
    dMatrix(unsigned _s) : vector<double>(_s*(_s-1)), rSize(_s) {}

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
    void printOn(ostream & _os)
    {
      unsigned index=0;
      for (unsigned i=0; i<rSize; i++)
	{
	  for (unsigned j=0; j<rSize; j++)
	    _os << this->operator[](index++) << " " ;
	  _os << endl;
	}
      _os << endl;
    }

    private:
      unsigned rSize;		   // row size (== number of columns!)
  };

    // private data of class eoSharing
private: 
  double nicheSize;
  eoQuadDistance<EOT> repDist;	   // default distance 
  eoDistance & dist;	     // allows to pass a specific distance
};

#endif
