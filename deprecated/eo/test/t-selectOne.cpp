// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// t-selectOne.cpp
//   This program test the breeder object
// (c) GeNeura Team, 1998
/*
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

#ifndef __GNUG__
// to avoid long name warnings
#pragma warning(disable:4786)
#endif // __GNUG__

#include <ga/eoBin.h>  // eoBin, eoPop, eoBreeder
#include <eoPop.h>
#include <ga/eoBitOp.h>

#include <eoUniformSelect.h>
#include <eoStochTournament.h>
#include <eoDetTournament.h>


//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

#include "binary_value.h"

//-----------------------------------------------------------------------------

main()
{
  const unsigned POP_SIZE = 8, CHROM_SIZE = 4;
  unsigned i;

  eoBinRandom<Chrom> random;
  eoPop<Chrom> pop;

  // Create the population
  for (i = 0; i < POP_SIZE; ++i) {
    Chrom chrom(CHROM_SIZE);
    random(chrom);
    BinaryValue()(chrom);
    pop.push_back(chrom);
  }

  // print population
  std::cout << "population:" << std::endl;
  for (i = 0; i < pop.size(); ++i)
    std::cout << pop[i] << " " << pop[i].fitness() << std::endl;

  // Declare 1-selectors
  eoUniformSelect<Chrom> uSelect;

  Chrom aChrom;
  aChrom = uSelect( pop );
  std::cout << "Uniform Select " << aChrom << " " << aChrom.fitness() << std::endl;

  eoStochTournament<Chrom> sSelect(0.7);
  aChrom = sSelect( pop );
  std::cout << "Stochastic Tournament " << aChrom << " " << aChrom.fitness() << std::endl;

  eoDetTournament<Chrom> dSelect(3);
  aChrom = dSelect( pop );
  std::cout << "Deterministic Tournament " << aChrom << " " << aChrom.fitness() << std::endl;

  return 0;
}

//-----------------------------------------------------------------------------
