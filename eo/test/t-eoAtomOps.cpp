/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoAtomOps.cpp
      Program that tests the atomic operator classes

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
#include <other/eoString.h>

// RNGs
#include <eoNegExp.h>

main(int argc, char *argv[]) {
  eoString<float> aString("123456");
  eoAtomCreep<char> creeper;
  eoAtomMutation< eoString<float> > mutator( creeper, 0.5 );

  eoNegExp<char> charNE( 2 );
  eoAtomRandom<char> randomer( charNE );
  eoAtomMutation<  eoString<float> > mutator2 ( randomer, 0.5 );

  std::cout << "Before aString " << aString << std::endl;
  mutator( aString);
  std::cout << " after mutator " << aString << std::endl;
  mutator2( aString);
  std::cout << " after mutator2 " << aString << std::endl;;
  return 0; // to avoid VC++ complaints
}

