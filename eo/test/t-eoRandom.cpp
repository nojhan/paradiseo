/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoRandom.cpp
      Test program for random generator

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

/**
CVS Info: $Date: 2001-02-18 04:34:57 $  $Author: evomarc $ $Revision: 1.9 $
*/

//-----------------------------------------------------------------------------

#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream
#include <utils/eoRndGenerators.h>         // eoBin
//#include <eoNormal.h>
//#include <eoNegExp.h>

//-----------------------------------------------------------------------------

main() {
  try{
  eoUniformGenerator<float> u1(-2.5,3.5);
  eoUniformGenerator<double> u2(0.0003, 0.0005 );
  eoUniformGenerator<unsigned long> u3( 100000U, 10000000U);

  eoNegExpGenerator<float> e1(3.5);
  eoNegExpGenerator<double> e2(0.003 );
  eoNegExpGenerator<long> e3( 10000U);
  /*  cout << "n1\t\tn2\t\tn3\t\te1\t\te2\t\te3" << endl; */
  for ( unsigned i = 0; i < 100; i ++) {
    cout << "Uniform: " << u1() << "\t" << u2() << "\t" << u3() << endl;
    cout << "NegExp: " << e1() << "\t" << e2() << "\t" << e3() << endl;
  }
  } 
 
  catch (exception& e)
    {
        cout << "exception: " << e.what() << endl;;
        exit(EXIT_FAILURE);
    }
  return 0; // to avoid VC++ complaints
}

//-----------------------------------------------------------------------------
