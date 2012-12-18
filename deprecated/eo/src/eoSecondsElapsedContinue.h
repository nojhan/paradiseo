// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSeconsElapsedContinue.h
// (c) Maarten Keijzer, 2007
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

#ifndef _eoSecondsElapsedContinue_h
#define _eoSecondsElapsedContinue_h

#include <eoContinue.h>

/**
    Timed continuator: continues until a number of seconds is used

    @ingroup Continuators
*/
template< class EOT>
class eoSecondsElapsedContinue: public eoContinue<EOT>
{
    time_t start;
    int seconds;
public:

  eoSecondsElapsedContinue(int nSeconds) : start(time(0)), seconds(nSeconds) {}

  virtual bool operator() ( const eoPop<EOT>& _vEO ) {
        time_t now = time(0);
        time_t diff = now - start;

        if (diff >= seconds) return false; // stop
        return true;

    }


  virtual std::string className(void) const { return "eoSecondsElapsedContinue"; }

  /** REad from a stream
   * @param __is the ostream to read from
   */
  void readFrom (std :: istream & __is) {

    __is >> start >> seconds;
  }

  /** Print on a stream
   * @param __os the ostream to print on
   */
  void printOn (std :: ostream & __os) const {

    __os << start << ' ' << seconds << std :: endl;
  }

};
/** @example t-eoSecondsElapsedContinue.cpp
 */

#endif
