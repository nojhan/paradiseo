//-----------------------------------------------------------------------------
// t-eoMGE.cpp
//-----------------------------------------------------------------------------

#ifndef __GNUG__
// to avoid long name warnings
#pragma warning(disable:4786)
#endif // __GNUG__

#include <eo>
#include <ga/eoBitOp.h>

#include "binary_value.h"

// Viri
#include <MGE/VirusOp.h>
#include <MGE/eoVirus.h>
#include <MGE/eoInitVirus.h>

//-----------------------------------------------------------------------------

typedef eoVirus<float> Chrom;

//-----------------------------------------------------------------------------

int main()
{
  const unsigned POP_SIZE = 100, CHROM_SIZE = 16;
  unsigned i;
  eoBooleanGenerator gen;

  // the populations: 
  eoPop<Chrom> pop; 

   // Evaluation
  eoEvalFuncPtr<Chrom> eval(  binary_value );
  
  eoInitVirus<float> random(CHROM_SIZE, gen); 
  for (i = 0; i < POP_SIZE; ++i) {
      Chrom chrom;
      random(chrom);
      eval(chrom);
      pop.push_back(chrom);
  }
  
  cout << "population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << "\t" << pop[i] << " " << pop[i].fitness() << endl;

  
  // selection
  eoDetTournamentSelect<Chrom> lottery( 3) ;

  // breeder
  VirusMutation<float> vm;
  VirusTransmission<float> vt;
  VirusBitFlip<float> vf;
  eoUBitXover<Chrom> xover;
  eoProportionalOp<Chrom> propSel;
  eoGeneralBreeder<Chrom> breeder( lottery, propSel );
  propSel.add(vm, 0.25);
  propSel.add(vf, 0.25);
  propSel.add(vt, 0.25);
  propSel.add(xover, 0.25);
  
  // Replace a single one
  eoPlusReplacement<Chrom> replace;

  // Terminators
  eoGenContinue<Chrom> continuator1(50);
  eoFitContinue<Chrom> continuator2(65535.f);
  eoCombinedContinue<Chrom> continuator(continuator1, continuator2);  
  eoCheckPoint<Chrom> checkpoint(continuator);
  eoStdoutMonitor monitor;
  checkpoint.add(monitor);
  eoSecondMomentStats<Chrom> stats;
  monitor.add(stats);
  checkpoint.add(stats);

  // GA generation
  eoEasyEA<Chrom> ea(checkpoint, eval,  breeder, replace );

  // evolution
  try
    {
      ea(pop);
    }
  catch (exception& e)
    {
	cout << "exception: " << e.what() << endl;;
	exit(EXIT_FAILURE);
    }
  
  cout << "pop" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << "\t" <<  pop[i] << " " << pop[i].fitness() << endl;
  
  return 0;
}

//-----------------------------------------------------------------------------
