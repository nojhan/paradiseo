// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPopStat.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2001
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

/** WARNING: this file contains 2 classes:

eoPopString and eoSortedPopString

that transform the population into a string
that can be used to dump to the screen
*/

#ifndef _eoPopStat_h
#define _eoPopStat_h

#include <utils/eoStat.h>


/** Thanks to MS/VC++, eoParam mechanism is unable to handle vectors of stats.
This snippet is a workaround: 
This class will "print" a whole population into a string - that you can later
send to any stream
This is the plain version - see eoPopString for the Sorted version

Note: this Stat should probably be used only within eoStdOutMonitor, and not 
inside an eoFileMonitor, as the eoState construct will work much better there.
*/
template <class EOT>
class eoPopStat : public eoStat<EOT, string>
{
public :
  /** default Ctor, void string by default, as it appears 
      on the description line once at beginning of evolution. and
      is meaningless there. _howMany defaults to 0, that is, the whole
	  population*/
   eoPopStat(unsigned _howMany = 0, string _desc ="") 
	 : eoStat<EOT, string>("", _desc), combien( _howMany) {}
 
/** Fills the value() of the eoParam with the dump of the population.
Adds a \n before so it does not get mixed up with the rest of the stats
that are written by the monitor it is probably used from.
*/
void operator()(const eoPop<EOT>& _pop)
{
  char buffer[1023]; // about one K of space per member
  value() = "\n====== Pop dump =====\n";
  unsigned howMany=combien?combien:_pop.size();
  for (unsigned i = 0; i < howMany; ++i)
  {
      std::ostrstream os(buffer, 1022); // leave space for emergency terminate
      os << _pop[i] << endl << ends;
 
      // paranoid:
      buffer[1022] = '\0';
      value() += buffer;
  }
}
private:
  unsigned combien;
};

/** Thanks to MS/VC++, eoParam mechanism is unable to handle vectors of stats.
This snippet is a workaround: 
This class will "print" a whole population into a string - that you can later
send to any stream
This is the Sorted version - see eoPopString for the plain version

Note: this Stat should probably be used only within eoStdOutMonitor, and not 
inside an eoFileMonitor, as the eoState construct will work much better there.
*/
template <class EOT>
class eoSortedPopStat : public eoSortedStat<EOT, string>
{
public :
  /** default Ctor, void string by default, as it appears 
      on the description line once at beginning of evolution. and
      is meaningless there */
   eoSortedPopStat(string _desc ="") : eoSortedStat<EOT, string>("", _desc) {}
 
/** Fills the value() of the eoParam with the dump of the population.
Adds a \n before so it does not get mixed up with the rest of the stats
that are written by the monitor it is probably used from.
*/
void operator()(const vector<const EOT*>& _pop)
{
  char buffer[1023]; // about one K of space per member
  value() = "\n====== Pop dump =====\n";
  for (unsigned i = 0; i < _pop.size(); ++i)
  {
      std::ostrstream os(buffer, 1022); // leave space for emergency terminate
      os << *_pop[i] << endl << ends;
 
      // paranoid:
      buffer[1022] = '\0';
      value() += buffer;
  }
}
};

#endif
