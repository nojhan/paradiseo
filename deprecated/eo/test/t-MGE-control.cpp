//-----------------------------------------------------------------------------
// t-eoMGE.cpp
//-----------------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef __GNUG__
// to avoid long name warnings
#pragma warning(disable:4786)
#endif // __GNUG__

#include "eo"
#include "ga/eoBitOp.h"

#include "RoyalRoad.h"

// Viri
#include "VirusOp.h"
#include "eoVirus.h"
#include "eoInitVirus.h"

//-----------------------------------------------------------------------------

typedef eoVirus<float> Chrom;

//-----------------------------------------------------------------------------

int main()
{
  const unsigned POP_SIZE = 10, CHROM_SIZE = 12;
  unsigned i;
  eoBooleanGenerator gen;

  // the populations:
  eoPop<Chrom> pop;

  // Evaluation
  RoyalRoad<Chrom> rr( 8 );
  eoEvalFuncCounter<Chrom> eval( rr );

  eoInitVirus<float> random(CHROM_SIZE, gen);
  for (i = 0; i < POP_SIZE; ++i) {
      Chrom chrom;
      random(chrom);
      eval(chrom);
      pop.push_back(chrom);
  }

  std::cout << "population:" << std::endl;
  for (i = 0; i < pop.size(); ++i)
    std::cout << "\t" << pop[i] << " " << pop[i].fitness() << std::endl;

  // selection
  eoStochTournamentSelect<Chrom> lottery(0.9 );

  // breeder
  eoOneBitFlip<Chrom> vm;
  eoUBitXover<Chrom> xover;
  eoProportionalOp<Chrom> propSel;
  eoGeneralBreeder<Chrom> breeder( lottery, propSel );
  propSel.add(vm, 0.2);
  propSel.add(xover, 0.8);

  // Replace a single one
  eoCommaReplacement<Chrom> replace;

  // Terminators
  eoGenContinue<Chrom> continuator1(10);
  eoFitContinue<Chrom> continuator2(CHROM_SIZE);
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
  eoEasyEA<Chrom> ea(checkpoint, eval,  breeder, replace );

  // evolution
  try
    {
      ea(pop);
    }
  catch (std::exception& e)
    {
	std::cout << "exception: " << e.what() << std::endl;;
	exit(EXIT_FAILURE);
    }

  std::cout << "pop" << std::endl;
  for (i = 0; i < pop.size(); ++i)
    std::cout << "\t" <<  pop[i] << " " << pop[i].fitness() << std::endl;

  std::cout << "\n --> Number of Evaluations = " << eval.getValue() << std::endl;
  return 0;
}

//-----------------------------------------------------------------------------
