/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

    t-eobin.cpp
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
#include <ga.h>	      // bitstring representation & operators
#include <utils/eoRndGenerators.h>
#include "binary_value.h"

//-----------------------------------------------------------------------------

typedef eoBit<float> Chrom;

//-----------------------------------------------------------------------------

void main_function()
{
  const unsigned SIZE = 8;
  unsigned i, j;
  eoBooleanGenerator gen;

  Chrom chrom(SIZE), chrom2;
  chrom.fitness(binary_value(chrom)); chrom2.fitness(binary_value(chrom2));

  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = true; chrom.fitness(binary_value(chrom));
  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = false; chrom.fitness(binary_value(chrom));
  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = true; chrom.fitness(binary_value(chrom));

  cout << "chrom.className() = " << chrom.className() << endl;

  cout << "chrom:  " << chrom << endl
       << "chrom2: " << chrom2 << endl;
  
  char buff[1024];
  ostrstream os(buff, 1024);
  os << chrom;
  istrstream is(os.str());
  is >> chrom2; chrom.fitness(binary_value(chrom2));
  
  cout << "\nTesting reading, writing\n";
  cout << "chrom:  " << chrom << "\nchrom2: " << chrom2 << '\n';
  
  fill(chrom.begin(), chrom.end(), false);
  cout << "--------------------------------------------------"
       << endl << "eoMonOp's aplied to .......... " << chrom << endl;

  eoInitFixedLength<Chrom>
      random(chrom.size(), gen);

  random(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBinRandom ............ " << chrom << endl;

  eoOneBitFlip<Chrom> bitflip;
  bitflip(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBitFlip .............. " << chrom << endl;
  
  eoBitMutation<Chrom> mutation(0.5);
  mutation(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBinMutation(0.5) ..... " << chrom << endl;

  eoBitInversion<Chrom> inversion;
  inversion(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBinInversion ......... " << chrom << endl;

  eoBitNext<Chrom> next;
  next(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBinNext .............. " << chrom << endl;

  eoBitPrev<Chrom> prev;
  prev(chrom); chrom.fitness(binary_value(chrom));
  cout << "after eoBinPrev .............. " << chrom << endl;

  fill(chrom.begin(), chrom.end(), false); chrom.fitness(binary_value(chrom));
  fill(chrom2.begin(), chrom2.end(), true); chrom2.fitness(binary_value(chrom2));
  cout << "--------------------------------------------------"
       << endl << "eoBinOp's aplied to ... " 
       << chrom << " " << chrom2 << endl;

  eo1PtBitXover<Chrom> xover;
  fill(chrom.begin(), chrom.end(), false);
  fill(chrom2.begin(), chrom2.end(), true);
  xover(chrom, chrom2); 
  chrom.fitness(binary_value(chrom)); chrom2.fitness(binary_value(chrom2));
  cout << "eoBinCrossover ........ " << chrom << " " << chrom2 << endl; 

  for (i = 1; i < SIZE; i++)
    {
      eoNPtsBitXover<Chrom> nxover(i);
      fill(chrom.begin(), chrom.end(), false);
      fill(chrom2.begin(), chrom2.end(), true);
      nxover(chrom, chrom2);
      chrom.fitness(binary_value(chrom)); chrom2.fitness(binary_value(chrom2));
      cout << "eoBinNxOver(" << i << ") ........ " 
	   << chrom << " " << chrom2 << endl; 
    }

  for (i = 1; i < SIZE / 2; i++)
    for (j = 1; j < SIZE / 2; j++)
      {
	eoBitGxOver<Chrom> gxover(i, j);
	fill(chrom.begin(), chrom.end(), false);
	fill(chrom2.begin(), chrom2.end(), true);
	gxover(chrom, chrom2);
	chrom.fitness(binary_value(chrom)); chrom2.fitness(binary_value(chrom2));
	cout  << "eoBinGxOver(" << i << ", " << j << ") ..... " 
	      << chrom << " " << chrom2 << endl; 
      }

    // test SGA algorithm
    eoGenContinue<Chrom> continuator1(50);
    eoFitContinue<Chrom> continuator2(65535.f);

    eoCombinedContinue<Chrom> continuator(continuator1, continuator2);

    eoCheckPoint<Chrom> checkpoint(continuator);

    eoStdoutMonitor monitor;

    checkpoint.add(monitor);

    eoSecondMomentStats<Chrom> stats;

    monitor.add(stats);
    checkpoint.add(stats);

    eoProportionalSelect<Chrom> select;
    eoEvalFuncPtr<Chrom>  eval(binary_value);

    eoSGA<Chrom> sga(select, xover, 0.8f, bitflip, 0.1f, eval, checkpoint);

    eoInitFixedLength<Chrom> init(16, gen);
    eoPop<Chrom> pop(100, init);

    apply<Chrom>(eval, pop);

    sga(pop);

    pop.sort();

    cout << "Population " << pop << endl;

    cout << "\nBest: " << pop[0].fitness() << '\n';

  /*

    Commented this out, waiting for a definite decision what to do with the mOp's

    // Check multiOps
    eoMultiMonOp<Chrom> mOp( &next );
    mOp.adOp( &bitflip );
    cout << "before multiMonOp............  " << chrom << endl;
    mOp( chrom );
    cout << "after multiMonOp .............. " << chrom << endl;

    eoBinGxOver<Chrom> gxover(2, 4);
    eoMultiBinOp<Chrom> mbOp( &gxover );
    mOp.adOp( &bitflip );
    cout << "before multiBinOp............  " << chrom << " " << chrom2 << endl;
    mbOp( chrom, chrom2 );
    cout << "after multiBinOp .............. " << chrom << " " << chrom2 <<endl;
  */
}

//-----------------------------------------------------------------------------
// For MSVC memory lead detection
#ifdef _MSC_VER
#include <crtdbg.h> 
#endif

int main()
{
#ifdef _MSC_VER
  //  rng.reseed(42);
    int flag = _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);
     flag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag(flag);
//   _CrtSetBreakAlloc(100);
#endif

    try
    {
        main_function();
    }
    catch(exception& e)
    {
        cout << "Exception: " << e.what() << '\n';
    }

}
