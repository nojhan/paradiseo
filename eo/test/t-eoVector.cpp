/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoVectpr.cpp
      This program tests vector-like chromosomes
    (c) GeNeura Team, 1999, 2000
 
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
//-----------------------------------------------------------------------------

#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream

#include <eoUniform.h>
#include <eoVector.h>         // eoVector
#include <eo1dWDistance.h>

//-----------------------------------------------------------------------------

typedef eoVector<float> Chrom;

//-----------------------------------------------------------------------------

main()
{
  const unsigned SIZE = 4;

  eoUniform<Chrom::Type> uniform(-1,1);

  Chrom chrom1(SIZE,uniform), chrom2( SIZE, uniform);
  
  cout << "chrom1:  " << chrom1 << endl <<
    "chrom2:  " << chrom2 << endl;
  
  eo1dWDistance< float, float > chromDist( chrom1 );
  cout << "Distance from chrom1 to chrom2 " << chromDist.distance( chrom2 ) << endl;
  
  
  return 0;
}

//-----------------------------------------------------------------------------
