//-----------------------------------------------------------------------------
// t-eoMGE.cpp
//-----------------------------------------------------------------------------

#ifndef __GNUG__
// to avoid long name warnings
#pragma warning(disable:4786)
#endif // __GNUG__

#include <eo>
#include <ga/eoBitOp.h>

#include "RoyalRoad.h"

// Viri
#include <MGE/VirusOp.h>
#include <MGE/eoVirus.h>
#include <MGE/eoInitVirus.h>

//-----------------------------------------------------------------------------

typedef eoVirus<float> Chrom;

//-----------------------------------------------------------------------------

int main()
{
  const unsigned POP_SIZE = 1000, CHROM_SIZE = 128;
  unsigned i;
  eoBooleanGenerator gen;

  // the populations: 
  eoPop<Chrom> pop; 

  // Evaluation
 RoyalRoad<Chrom> rr( 4 ); 
  
  eoInitVirus<float> random(CHROM_SIZE, gen); 
  for (i = 0; i < POP_SIZE; ++i) {
      Chrom chrom;
      random(chrom);
      rr(chrom);
      pop.push_back(chrom);
  }
  
  cout << "population:" << endl;
  for (i = 0; i < pop.size(); ++i)
    cout << "\t" << pop[i] << " " << pop[i].fitness() << endl;

  
  // selection
  eoStochTournamentSelect<Chrom> lottery(0.9 );

  // breeder
  VirusMutation<float> vm;
  VirusTransmission<float> vt;
  VirusBitFlip<float> vf;
  eoUBitXover<Chrom> xover;
  eoProportionalOp<Chrom> propSel;
  eoGeneralBreeder<Chrom> breeder( lottery, propSel );
  propSel.add(vm, 0.4);
  propSel.add(vf, 0.4);
  propSel.add(vt, 0.1);
  propSel.add(xover, 0.1);
  
  // Replace a single one
  eoCommaReplacement<Chrom> replace;

  // Terminators
  eoGenContinue<Chrom> continuator1(500);
  eoFitContinue<Chrom> continuator2(128);
  eoCombinedContinue<Chrom> continuator(continuator1, continuator2);  
  eoCheckPoint<Chrom> checkpoint(continuator);
  eoStdoutMonitor monitor;
  checkpoint.add(monitor);
  eoSecondMomentStats<Chrom> stats;
  eoPopStat<Chrom> dumper( 10 );
  monitor.add(stats);
  checkpoint.add(dumper);
  checkpoint.add(stats);

  // GA generation
  eoEasyEA<Chrom> ea(checkpoint, rr,  breeder, replace );

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
