/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoGeneralOps.cpp
      Program that tests the General operator interface, and the wrappers
      for monary and unary operators.

    (c) GeNeura Team, 1999 
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 
    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
            
*/

//-----------------------------------------------------------------------------// 

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <string>
#include <iostream>
#include <iterator>

using namespace std;

// Operators we are going to test
#include <eoAtomCreep.h>
#include <eoAtomBitFlip.h>
#include <eoAtomRandom.h>
#include <eoAtomMutation.h>

// Several EOs
#include <eoString.h>

// generalOp we are testing
#include <eoGeneralOp.h>

main(int argc, char *argv[]) {
  eoString<float> aString("123456");
  eoAtomCreep<char> creeper;
  eoAtomMutation< eoString<float> > mutator( creeper, 0.5 );

  eoWrappedMonOp< eoString<float> > wCreeper( mutator );
  cout << "Before aString " << aString;
  mutator( aString);
  cout << " after mutator " << aString;

  // Test now the alternative interface
  eoPop< eoString<float> > vIn, vOut;
  insert_iterator<eoPop<eoString<float> > > ins( vOut, vOut.begin() );
  vIn.push_back( aString );
  wCreeper( vIn.begin(), ins );

  cout << endl << "Before " << vIn[0] << endl << " after " << vOut[0] << endl;;
  
  return 0; // to avoid VC++ complaints
}

