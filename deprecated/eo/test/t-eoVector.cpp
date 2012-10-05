/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoVector.cpp
      This program tests vector-like chromosomes
    (c) GeNeura Team, 1999, 2000

    Modified by Maarten Keijzer 2001

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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <cassert>
#include <iostream>

#include <utils/eoRndGenerators.h>
#include <eoVector.h>         // eoVector
#include <eoInit.h>
#include <eoScalarFitness.h>

//-----------------------------------------------------------------------------

typedef eoVector<eoMaximizingFitness, int> Chrom1;
typedef eoVector<eoMinimizingFitness, int> Chrom2;

//-----------------------------------------------------------------------------

int main()
{
  const unsigned SIZE = 4;

  // check if the appropriate ctor gets called
  Chrom1 chrom(SIZE, 5);

  for (unsigned i = 0; i < chrom.size(); ++i)
  {
    assert(chrom[i] == 5);
  }

  eoUniformGenerator<Chrom1::AtomType> uniform(-1,1);
  eoInitFixedLength<Chrom1> init(SIZE, uniform);

  init(chrom);

  std::cout << chrom << std::endl;

  Chrom2 chrom2(chrom);

  std::cout << chrom2 << std::endl;

//  eoInitVariableLength<Chrom1> initvar(

  return 0;
}

//-----------------------------------------------------------------------------
