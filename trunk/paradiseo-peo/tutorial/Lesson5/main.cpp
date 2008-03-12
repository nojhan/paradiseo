#include <peo>
#include <peoAsyncDataTransfer.h>
#include <peoSyncDataTransfer.h>
#define SIZE 10
#define DEF_DOMAIN 100
#define POP_SIZE 100
#define SELECT_RATE 0.8
#define NB_GEN 1
#define XOVER_P 0.75
#define MUT_P 0.05
#define MIGRATIONS_AT_N_GENERATIONS 5
#define NUMBER_OF_MIGRANTS 10

struct Representation : public eoVector< eoMinimizingFitness, double >
  {

    Representation()
    {
      resize( SIZE );
    }
  };

struct Init : public eoInit< Representation >
  {

    void operator()( Representation& rep )
    {

      for ( int i = 0; i < SIZE; i++ )
        {
          rep[ i ] = (rng.uniform() - 0.5) * DEF_DOMAIN;
        }
    }
  };

struct Eval : public eoEvalFunc< Representation >
  {

    void operator()( Representation& rep )
    {

      double fitnessValue = 0.0;
      for ( int i = 0; i < SIZE; i++ )
        {

          fitnessValue += pow( rep[ i ], 2.0 );
        }

      rep.fitness( fitnessValue );
    }
  };

struct MutationOp : public eoMonOp< Representation >
  {

    bool operator()( Representation& rep )
    {

      unsigned int pos = (unsigned int)( rng.uniform() * SIZE );
      rep[ pos ] = (rng.uniform() - 0.5) * DEF_DOMAIN;

      rep.invalidate();

      return true;
    }
  };

struct XoverOp : public eoQuadOp< Representation >
  {

    bool operator()( Representation& repA, Representation& repB )
    {

      static Representation offA, offB;
      double lambda = rng.uniform();

      for ( int i = 0; i < SIZE; i++ )
        {

          offA[ i ] = lambda * repA[ i ] + ( 1.0 - lambda ) * repB[ i ];
          offB[ i ] = lambda * repB[ i ] + ( 1.0 - lambda ) * repA[ i ];
        }

      repA = offA;
      repB = offB;

      repA.invalidate();
      repB.invalidate();

      return true;
    }
  };



void pack( const Representation& rep )
{

  if ( rep.invalid() ) ::pack( (unsigned int)0 );
  else
    {
      ::pack( (unsigned int)1 );
      ::pack( (double)(rep.fitness()) );
    }

  for ( unsigned int index = 0; index < SIZE; index++ )
    {
      ::pack( (double)rep[ index ] );
    }
}

void unpack( Representation& rep )
{

  eoScalarFitness<double, std::greater<double> > fitness;
  unsigned int validFitness;

  unpack( validFitness );
  if ( validFitness )
    {

      double fitnessValue;
      ::unpack( fitnessValue );
      rep.fitness( fitnessValue );
    }
  else
    {
      rep.invalidate();
    }

  double value;
  for ( unsigned int index = 0; index < SIZE; index++ )
    {
      ::unpack( value );
      rep[ index ] = value;
    }
}

int main( int __argc, char** __argv )
{


  rng.reseed( time( NULL ) );
  srand( time( NULL ) );

  peo::init( __argc,  __argv );


  eoParser parser ( __argc, __argv );

  eoValueParam < unsigned int > nbGenerations( NB_GEN, "maxGen");
  parser.processParam ( nbGenerations );

  eoValueParam < double > selectionRate( SELECT_RATE, "select");
  parser.processParam ( selectionRate );


  RingTopology ring,topo;
  unsigned int dataA, dataB, dataC;

  dataA = 1;
  dataB = 5;
  dataC = 10;

  peoSyncDataTransfer dataTransfer( dataA, ring );
  peoSyncDataTransfer dataTransferb( dataB, ring );
  peoSyncDataTransfer dataTransferc( dataC, ring );


  Init init;
  Eval eval;
  eoPop< Representation > pop( POP_SIZE, init );
  MutationOp mut;
  XoverOp xover;
  eoSGATransform< Representation > transform( xover, XOVER_P, mut, MUT_P );
  eoStochTournamentSelect< Representation > select;
  eoSelectMany< Representation > selectN( select, selectionRate.value() );
  eoSSGAStochTournamentReplacement< Representation > replace( 1.0 );
  eoWeakElitistReplacement< Representation > elitReplace( replace );
  eoGenContinue< Representation > cont( nbGenerations.value() );
  eoCheckPoint< Representation > checkpoint( cont );
  eoEasyEA< Representation > algo( checkpoint, eval, selectN, transform, elitReplace );



  //-------------------------------------------------------------------------------------------------------------
  // MIGRATION CONTEXT DEFINITION

  eoPeriodicContinue< Representation > mig_conti(  MIGRATIONS_AT_N_GENERATIONS );
  eoContinuator<Representation> mig_cont(mig_conti,pop);
  eoRandomSelect<Representation> mig_select_one;
  eoSelector <Representation, eoPop<Representation> > mig_select (mig_select_one,NUMBER_OF_MIGRANTS,pop);
  eoPlusReplacement<Representation> replace_one;
  eoReplace <Representation, eoPop<Representation> > mig_replace (replace_one,pop);
//  peoSyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig(MIGRATIONS_AT_N_GENERATIONS,mig_select,mig_replace,topo);
  peoAsyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig(mig_cont,mig_select,mig_replace,topo);
  checkpoint.add( mig );
  //-------------------------------------------------------------------------------------------------------------

  eoPop< Representation > pop2( POP_SIZE, init );
  eoSGATransform< Representation > transform2( xover, XOVER_P, mut, MUT_P );
  eoStochTournamentSelect< Representation > select2;
  eoSelectMany< Representation > selectN2( select2, selectionRate.value() );
  eoSSGAStochTournamentReplacement< Representation > replace2( 1.0 );
  eoWeakElitistReplacement< Representation > elitReplace2( replace2 );
  eoGenContinue< Representation > cont2( nbGenerations.value() );
  eoCheckPoint< Representation > checkpoint2( cont2 );
  eoEasyEA< Representation > algo2( checkpoint2, eval, selectN2, transform2, elitReplace2 );

  //-------------------------------------------------------------------------------------------------------------
  // MIGRATION CONTEXT DEFINITION

  eoPeriodicContinue< Representation > mig_conti2(  MIGRATIONS_AT_N_GENERATIONS );
  eoContinuator<Representation> mig_cont2(mig_conti2,pop2);
  eoRandomSelect<Representation> mig_select_one2;
  eoSelector <Representation, eoPop<Representation> > mig_select2 (mig_select_one2,NUMBER_OF_MIGRANTS,pop2);
  eoPlusReplacement<Representation> replace_one2;
  eoReplace <Representation, eoPop<Representation> > mig_replace2 (replace_one2,pop2);
//  peoSyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig2(MIGRATIONS_AT_N_GENERATIONS,mig_select2,mig_replace2,topo);
  peoAsyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig2(mig_cont2,mig_select2,mig_replace2,topo);
  checkpoint2.add( mig2 );
  //-------------------------------------------------------------------------------------------------------------

  eoPop< Representation > pop3( POP_SIZE, init );
  eoSGATransform< Representation > transform3( xover, XOVER_P, mut, MUT_P );
  eoStochTournamentSelect< Representation > select3;
  eoSelectMany< Representation > selectN3( select3, selectionRate.value() );
  eoSSGAStochTournamentReplacement< Representation > replace3( 1.0 );
  eoWeakElitistReplacement< Representation > elitReplace3( replace3 );
  eoGenContinue< Representation > cont3( nbGenerations.value() );
  eoCheckPoint< Representation > checkpoint3( cont3 );
  eoEasyEA< Representation > algo3( checkpoint3, eval, selectN3, transform3, elitReplace3 );

  //-------------------------------------------------------------------------------------------------------------
  // MIGRATION CONTEXT DEFINITION

  eoPeriodicContinue< Representation > mig_conti3(  MIGRATIONS_AT_N_GENERATIONS );
  eoContinuator<Representation> mig_cont3(mig_conti3,pop3);
  eoRandomSelect<Representation> mig_select_one3;
  eoSelector <Representation, eoPop<Representation> > mig_select3 (mig_select_one3,NUMBER_OF_MIGRANTS,pop3);
  eoPlusReplacement<Representation> replace_one3;
  eoReplace <Representation, eoPop<Representation> > mig_replace3 (replace_one3,pop3);
//  peoSyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig3(MIGRATIONS_AT_N_GENERATIONS,mig_select3,mig_replace3,topo);
  peoAsyncIslandMig< eoPop< Representation >, eoPop< Representation > > mig3(mig_cont3,mig_select3,mig_replace3,topo);

  checkpoint3.add( mig3 );
  //-------------------------------------------------------------------------------------------------------------

  peoWrapper algoPar( algo, pop );
  mig.setOwner( algoPar );
  checkpoint.add( dataTransfer );
  dataTransfer.setOwner( algoPar );


  peoWrapper algoPar2( algo2, pop2 );
  mig2.setOwner( algoPar2 );
  checkpoint2.add( dataTransferb );
  dataTransferb.setOwner( algoPar2 );


  peoWrapper algoPar3( algo3, pop3 );
  mig3.setOwner( algoPar3 );
  checkpoint3.add( dataTransferc );
  dataTransferc.setOwner( algoPar3 );

  peo::run();
  peo::finalize();

  if ( getNodeRank() == 1 )
    std::cout << "A: " << dataA << std::endl;
  if ( getNodeRank() == 2 )
    std::cout << "B: " << dataB << std::endl;
  if ( getNodeRank() == 3 )
    std::cout << "C: " << dataC << std::endl;


  return 0;
}
