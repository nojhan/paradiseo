/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eogeneration.cpp
      Testing the eoGeneration classes, and classes related to it

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

//-----------------------------------------------------------------------------// 


// to avoid long name warnings
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <eoGeneration.h>
#include <eoEvalFuncPtrCnt.h>

#include "binary_value.h"

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

//-----------------------------------------------------------------------------

main()
{
  const unsigned POP_SIZE = 8, CHROM_SIZE = 16;
  unsigned i;

  eoUniform<Chrom::Type> uniform(false, true);
  eoBinRandom<Chrom> random;
  eoPop<Chrom> pop; 
  // Evaluation
  eoEvalFuncPtr<Chrom> eval( binary_value );

  for (i = 0; i < POP_SIZE; ++i)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      eval(chrom);
      pop.push_back(chrom);
    }
  
  cout << "population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << "\t" << pop[i] << " " << pop[i].fitness() << endl;

  
  // selection
  eoLottery<Chrom> lottery;

  // breeder
  eoBinBitFlip<Chrom> bitflip;
  eoBinCrossover<Chrom> xover;
  eoProportionalOpSel<Chrom> propSel;
  eoBreeder<Chrom> breeder( propSel );
  propSel.addOp(bitflip, 0.25);
  propSel.addOp(xover, 0.75);
  
  // replacement
  eoInclusion<Chrom> inclusion;

  

  // GA generation
  eoGeneration<Chrom> generation(lottery, breeder, inclusion, eval);

  // evolution
  unsigned g = 0;
  do {
    try
      {
	generation(pop);
      }
    catch (exception& e)
      {
	cout << "exception: " << e.what() << endl;;
	exit(EXIT_FAILURE);
      }
    
    cout << "pop[" << ++g << "]" << endl;
    for (i = 0; i < pop.size(); ++i)
      cout << "\t" <<  pop[i] << " " << pop[i].fitness() << endl;
    
  } while (pop[0].fitness() < pow(2.0, CHROM_SIZE) - 1);

  // Try again, with a "counted" evaluation function
  // GA generation
  // Evaluation
  eoEvalFuncPtrCnt<Chrom> eval2( binary_value );
  eoPop<Chrom> pop2; 
  
  for (i = 0; i < POP_SIZE; ++i)
    {
      Chrom chrom(CHROM_SIZE);
      random(chrom);
      binary_value(chrom);
      eval2(chrom);
      pop2.push_back(chrom);
    }
  eoGeneration<Chrom> generation2(lottery, breeder, inclusion, eval2);
  
  // evolution
  do {
    try
      {
	generation2(pop2);
      }
    catch (exception& e)
      {
	cout << "exception: " << e.what() << endl;;
	exit(EXIT_FAILURE);
      }
    
    cout << "pop[" << ++g << "]" << endl;
    for (i = 0; i < pop2.size(); ++i)
      cout << "\t" <<  pop2[i] << " " << pop[i].fitness() << endl;
    
  } while (pop2[0].fitness() < pow(2.0, CHROM_SIZE) - 1);

  cout << "Number of evaluations " << eval2.getNumOfEvaluations() << endl;
  return 0;
}

//-----------------------------------------------------------------------------
