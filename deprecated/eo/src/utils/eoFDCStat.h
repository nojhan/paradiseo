// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFDCStat.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000, 2001
/*
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
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoFDCStat_h
#define _eoFDCStat_h

#include <utils/eoStat.h>
#include <utils/eoDistance.h>
#include <utils/eoFileSnapshot.h>

/**
    The Fitness Distance Correlation computation.

    Stores the values into eoValueParam<EOT,double>
so they can be snapshot by some eoGnuplotSnapshot ...

@ingroup Stats
*/
template <class EOT>
class eoFDCStat : public eoStat<EOT, double>
{
public:

    using eoStat<EOT, double>::value;

    /** Ctor without the optimum */
  eoFDCStat(eoDistance<EOT> & _dist, std::string _description = "FDC") :
      eoStat<EOT,double>(0, _description), dist(_dist), boolOpt(false) {}

  /** Ctor with the optimum
   */
  eoFDCStat(eoDistance<EOT> & _dist, EOT & _theBest,
            std::string _description = "FDC") :
    eoStat<EOT,double>(0, _description), dist(_dist),
    theBest(_theBest), boolOpt(true) {}

  /** Compute the FDC - either from best in pop, or from absolute best
   *  if it was passed in the constructor
   */
    virtual void operator()(const eoPop<EOT>& _pop)
    {
      unsigned i;
      if (!boolOpt)                // take the local best
        theBest = _pop.best_element();
      unsigned int pSize = _pop.size();
      distToBest.value().resize(pSize);
      fitnesses.value().resize(pSize);
      double sumFit = 0.0, sumDist = 0.0;
      for (i=0; i<pSize; i++)
        {
          sumDist += (distToBest.value()[i] = dist(_pop[i], theBest));
          sumFit += (fitnesses.value()[i] = _pop[i].fitness());
        }
      // now the FDC coefficient
      double avgDist = sumDist/pSize;
      double avgFit = sumFit/pSize;
      sumDist = sumFit = 0.0;
      double num = 0.0;
      for (i=0; i<pSize; i++)
        {
          double devDist = distToBest.value()[i] - avgDist ;
          double devFit = fitnesses.value()[i] - avgFit ;
          sumDist += devDist*devDist;
          sumFit += devFit * devFit;
          num += devDist * devFit ;
        }
      value() = num/(sqrt(sumDist)*sqrt(sumFit));
    }

  /** accessors to the private eoValueParam<std::vector<double> >
   */
  const eoValueParam<std::vector<double> > & theDist()
  { return distToBest; }
  const eoValueParam<std::vector<double> > & theFit()
  { return fitnesses; }


private:
  eoDistance<EOT> & dist;
  EOT theBest;
  bool boolOpt;                    // whether the best is known or not
  eoValueParam<std::vector<double> > distToBest;
  eoValueParam<std::vector<double> > fitnesses;
};

/** Specific class for FDCStat monitoring:
 *  As I failed to have FDC stat as an eoStat, this is the trick
 *  to put the 2 eoParam<std::vector<double> > into a monitor
 *  This class does nothing else.

@ingroup Stats
 */
template <class EOT>
class eoFDCFileSnapshot : public eoFileSnapshot	// is an eoMonitor
{
public:
  /** Ctor: in addition to the parameters of the ctor of an eoFileSnapshot
            we need here an eoFDCStat. The 2 std::vectors (distances to optimum
            and fitnesses) are added to the monitor so they can be processed
            later to a file - and eventually by gnuplot
  */
  eoFDCFileSnapshot(eoFDCStat<EOT> & _FDCstat,
                    std::string _dirname = "tmpFDC", unsigned _frequency = 1,
                    std::string _filename = "FDC", std::string _delim = " "):
    eoFileSnapshot(_dirname, _frequency, _filename, _delim),
    FDCstat(_FDCstat)
  {
    eoFileSnapshot::add(FDCstat.theDist());
    eoFileSnapshot::add(FDCstat.theFit());
  }

  /** just to be sure the add method is not called further
   */
  virtual void add(const eoParam& _param)
    { throw std::runtime_error("eoFDCFileSnapshot::add(). Trying to add stats to an eoFDCFileSnapshot"); }

private:
  eoFDCStat<EOT> & FDCstat;
};

#endif
