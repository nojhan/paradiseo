// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*-----------------------------------------------------------------------------
 * t-eoRandom
 *    Testing program for the eoRNG class
 * (c) GeNeura Team, 1999 

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public
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


#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream
#include <eoUniform.h>         // eoBin
#include <eoNormal.h>
#include <eoNegExp.h>

//-----------------------------------------------------------------------------

main() {
  eoNormal<float> n1(-2.5,3.5);
  eoNormal<double> n2(0.003, 0.0005 );
  eoNormal<unsigned long> n3( 10000000U, 10000U);
  eoNegExp<float> e1(3.5);
  eoNegExp<double> e2(0.003 );
  eoNegExp<long> e3( 10000U);
  cout << "n1\t\tn2\t\tn3\t\te1\t\te2\t\te3" << endl;
  for ( unsigned i = 0; i < 100; i ++) {
    cout << n1() << "\t" << n2() << "\t" << n3() << "\t" <<
      e1() << "\t" << e2() << "\t" << e3() << endl;
  }
 
  return 0; // to avoid VC++ complaints
}

//-----------------------------------------------------------------------------
