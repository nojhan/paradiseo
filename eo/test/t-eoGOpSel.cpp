/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eoGOpSel.cpp
      Testing proportional operator selectors

    (c) Maarten Keijzer and GeNeura Team, 2000 
 
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

// to avoid long name warnings
#pragma warning(disable:4786)

#include <iostream>

#include <ga/eoBin.h>  // eoBin, eoPop, eoBreeder
#include <eoPop.h>
#include <ga/eoBitOp.h>
#include <eoProportionalGOpSel.h>
#include <eoSequentialGOpSelector.h>
#include <eoRandomIndiSelector.h>

#include <eoDetTournamentIndiSelector.h>
#include <eoDetTournamentInserter.h>
#include <eoStochTournamentInserter.h>

#include <eoGOpBreeder.h>

#include <utils/eoRNG.h>
#include <utils/eoState.h>

// Fitness evaluation
#include "binary_value.h"

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

//-----------------------------------------------------------------------------

main()
{
  rng.reseed(42); // reproducible random seed

  const unsigned POP_SIZE = 8, CHROM_SIZE = 4;
  unsigned i;

  eoUniform<Chrom::Type> uniform(false, true);
  eoBinRandom<Chrom> random;
  eoPop<Chrom> pop; 
  
  for (i = 0; i < POP_SIZE; ++i)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      chrom.fitness(binary_value(chrom));
      pop.push_back(chrom);
    }
  
  cout << "population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << pop[i] << " " << pop[i].fitness() << endl;

  eoBinBitFlip<Chrom> bitflip;
  eoBinCrossover<Chrom> xover;
  
  eoEvalFuncPtr<Chrom> eval(binary_value);

  //Create the proportional operator selector and add the 
  // two operators creatd above to it.

  eoProportionalGOpSel<Chrom > propSel;
  eoSequentialGOpSel<Chrom>    seqSel;

  propSel.addOp(bitflip, 0.5);
  propSel.addOp(xover, 0.5);
  
  // seqSel selects operator in sequence, creating a combined operator
  // add a bitflip, an xover and another bitflip
  seqSel.addOp(bitflip, 0.25);
  seqSel.addOp(xover, 0.5);
  seqSel.addOp(bitflip, 0.25);


  eoRandomIndiSelector<Chrom>         selector1;
  eoDetTournamentIndiSelector<Chrom>  selector2(2);

  eoBackInserter<Chrom> inserter1;
  eoDetTournamentInserter<Chrom> inserter2(eval, 2);
  eoStochTournamentInserter<Chrom> inserter3(eval, 0.9f);

  eoGOpBreeder<Chrom> breeder1(propSel, selector1);
  eoGOpBreeder<Chrom> breeder2(seqSel, selector1);
  eoGOpBreeder<Chrom> breeder3(propSel, selector2);
  eoGOpBreeder<Chrom> breeder4(seqSel, selector2);

  // test the breeders

  breeder1(pop);
  breeder2(pop);
  breeder3(pop);
  breeder4(pop);

  eoState state;

  state.registerObject(pop);

  state.save(std::cout);
    
  return 0;
}

//-----------------------------------------------------------------------------
