// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsStdevXOver.h
// (c) GeNeura Team, 2000 - Maarten Keijzer 2000 - Marc Schoenauer 2001
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoEsStdevXOver_h
#define _eoEsStdevXOver_h

#include <es/eoEsStdev.h>
#include <es/eoGenericRealOp.h>

/**
\ingroup EvolutionStrategies

Crossover for Evolutionary strategie style representation,
    supporting co-evolving standard deviations.

Simply calls a crossover for the object variables,
    and a crossover for teh StdDev
*/

template <class EOT>
class eoEsStdevXOver : public eoQuadOp<EOT>
{
public :
  eoEsStdevXOver(eoQuadOp<vector<double> > & _objectXOver,
		 eoQuadOp<vector<double> > & _stdDevXOver) :
    objectXOver(_objectXOver), stdDevXOver(_stdDevXOver) {}

  std::string className(void) const { return "eoEsStdevXOver"; }

  bool operator()(EOT & _eo1, EOT & _eo2)
  {
    bool objectChanged = objectXOver(_eo1, _eo2); // as vector<double>
    bool stdDevChanged = stdDevXOver(_eo1.stdevs, _eo2.stdevs);

    /// Marc, I didn't change it, but if only the stdev has changed,
    /// doesn't that mean that the fitness is stil valid. Maarten
    if ( objectChanged || stdDevChanged )
      {
        return true;
      }

    return false;
  }

private:
  eoQuadOp<vector<double> > & objectXOver;
  eoQuadOp<vector<double> > & stdDevXOver;
};

/* A question: it seems it really makes no difference to have 
     as template the fitness (and use eoEsStdev<FitT> where you need EOs) or
     directly the EOT itself. But of course if the EOT you use does not have
     a stdev public member the compiler will crash.
   There is a difference, however, in the calling instruction, because in 
     on case you have to write eoEsStdevXOver<double> whereas otherwise you
     simply write eoEsStdevXOver<Indi> (if Indi has been typedef'ed correctly).
   So to keep everything (or almost :-) in the main program templatized 
     with the Indi i've kept here the EOT template. 

Are there arguments against that???
MS - Marc.Schoenauer@polytechnique.fr
*/

#endif
