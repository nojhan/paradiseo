/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoVirus.cpp
      This program tests the the binary cromosomes and several genetic operators
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
//-----------------------------------------------------------------------------

#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream
#include <eo>         // general EO
#include <MGE/eoVirus.h>  // bitstring representation & operators
#include <MGE/VirusOp.h>
#include <utils/eoRndGenerators.h>

#include "binary_value.h"

//-----------------------------------------------------------------------------

typedef eoVirus<float> Chrom;

//-----------------------------------------------------------------------------

int main()
{
  const unsigned SIZE = 8;
  eoBooleanGenerator gen;
  eo::rng.reseed( time( 0 ) );

  Chrom chrom(SIZE), chrom2(SIZE);
  chrom.fitness(binary_value(chrom)); chrom2.fitness(binary_value(chrom2));
  cout << chrom << endl;
  cout << chrom2 << endl;

  // Virus Mutation
  VirusBitFlip<float> vf;
  unsigned i;
  for ( i = 0; i < 10; i++ ) {
	vf( chrom );
	cout << chrom << endl;
  }

  // Chrom Mutation
  cout << "Chrom mutation--------" << endl;
  VirusMutation<float> vm;
  for ( i = 0; i < 10; i++ ) {
	vm( chrom );
	cout << chrom << endl;
  }

  // Chrom Transmision
  cout << "Chrom transmission--------" << endl;
  VirusTransmission<float> vt; 
  vt( chrom2, chrom );
  cout << chrom2 << endl;

  return 0;

}

