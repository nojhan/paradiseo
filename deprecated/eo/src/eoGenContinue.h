// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenContinue.h
// (c) GeNeura Team, 1999
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
 */
//-----------------------------------------------------------------------------

#ifndef _eoGenContinue_h
#define _eoGenContinue_h

#include <eoContinue.h>
#include <utils/eoParam.h>
#include <utils/eoLogger.h>

/**
    Generational continuator: continues until a number of generations is reached

    @ingroup Continuators
*/
template< class EOT>
class eoGenContinue: public eoContinue<EOT>, public eoValueParam<unsigned>
{
public:

  /// Ctor for setting a
  eoGenContinue( unsigned long _totalGens)
          : eoValueParam<unsigned>(0, "Generations", "Generations"),
            repTotalGenerations( _totalGens ),
            thisGenerationPlaceHolder(0),
            thisGeneration(thisGenerationPlaceHolder)
    {};

  /// Ctor for enabling the save/load the no. of generations counted
  eoGenContinue( unsigned long _totalGens, unsigned long& _currentGen)
          : eoValueParam<unsigned>(0, "Generations", "Generations"),
            repTotalGenerations( _totalGens ),
            thisGenerationPlaceHolder(0),
            thisGeneration(_currentGen)
    {};

  /** Returns false when a certain number of generations is
         * reached */
  virtual bool operator() ( const eoPop<EOT>& _vEO ) {
      (void)_vEO;
    thisGeneration++;
    value() = thisGeneration;

    if (thisGeneration >= repTotalGenerations)
      {
            eo::log << eo::logging << "STOP in eoGenContinue: Reached maximum number of generations [" << thisGeneration << "/" << repTotalGenerations << "]\n";
        return false;
      }
    return true;
  }

  /** Sets the number of generations to reach
            and sets the current generation to 0 (the begin)

        @todo replace this by an "init" method
    */
  virtual void totalGenerations( unsigned long _tg ) {
          repTotalGenerations = _tg;
          thisGeneration = 0;
        };

  /** Returns the number of generations to reach*/
  virtual unsigned long totalGenerations( )
  {
    return repTotalGenerations;
  };


  virtual std::string className(void) const { return "eoGenContinue"; }

  /** Read from a stream
   * @param __is the istream to read from
   */
  void readFrom (std :: istream & __is) {

    __is >> thisGeneration; /* Loading the number of generations counted */
  }

  /** Print on a stream
   * @param __os the ostream to print on
   */
  void printOn (std :: ostream & __os) const {

    __os << thisGeneration << std :: endl; /* Saving the number of generations counted */
  }

private:
  unsigned long repTotalGenerations;
  unsigned long thisGenerationPlaceHolder;
  unsigned long& thisGeneration;
};

#endif
