//eo
#include <eo>

// moeo
#include <moeo>

//general
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>

//peo
#include <peo>
#include <peoAsyncDataTransfer.h>
#include <peoSyncDataTransfer.h>

#include <dmls.h>


template <class A1, class A2, class R>
class generalAggregation: public eoBF<A1, A2, R>{};

template <class EOT, class A, class R>
  class AbstractEntityToPopAggregation: public generalAggregation<eoPop<EOT>&, A, R>{};

template <class EOT, class A, class R>
  class AbstractEntityToArchiveAggregation: public generalAggregation<moeoArchive<EOT>&, A, R>{};

template <class EOT>
  class PopToPopAggregation: public AbstractEntityToPopAggregation<EOT, eoPop<EOT>&, void>{};

template <class EOT>
  class ArchiveToPopAggregation: public AbstractEntityToPopAggregation<EOT, moeoArchive<EOT>&, void>{};

template <class EOT>
  class PopToArchiveAggregation: public AbstractEntityToArchiveAggregation<EOT, eoPop<EOT>&, void>{};

template <class EOT>
  class ArchiveToArchiveAggregation: public AbstractEntityToArchiveAggregation<EOT, moeoArchive<EOT>&, void>{};


/*
 *
 */
template <class EOT>
class gestionTransfer: public eoUpdater{

public:
  gestionTransfer(eoGenContinue < EOT > & _continuator, 
		  peoDataTransfer & _asyncDataTransfer,
		  moeoArchive<EOT>& _archive):continuator(_continuator), asyncDataTransfer(_asyncDataTransfer), archive(_archive){}

  void operator()(){
    if(!continuator(archive)){
      asyncDataTransfer();
      continuator.totalGenerations(continuator.totalGenerations());
    }
  }

private:
  eoGenContinue < EOT > & continuator;
  peoDataTransfer& asyncDataTransfer;
  moeoArchive<EOT>& archive;
};


/*
 *
 */
template <class EOT>
class gestionTransfer2: public eoUpdater{

public:
  gestionTransfer2(eoContinue < EOT > & _continuator, 
		  peoSyncDataTransfer & _asyncDataTransfer,
		  moeoArchive<EOT>& _archive):continuator(_continuator), asyncDataTransfer(_asyncDataTransfer), archive(_archive){}

  void operator()(){
    if(!continuator(archive)){
      asyncDataTransfer();
    }
  }

private:
  eoContinue < EOT > & continuator;
  peoSyncDataTransfer& asyncDataTransfer;
  moeoArchive<EOT>& archive;
};


/*
 *
 */
/*template <class EOT>
class IBEAAggregationRnd: public ArchiveToPopAggregation<EOT>{

public:
  void operator()(eoPop< EOT >& _pop, moeoArchive< EOT >& _archive) {
    UF_random_generator<unsigned int> rndGen;    
    std::vector<unsigned int> tab;
    unsigned int resizeValue;
    unsigned int popCorrectSize= _pop.size();

    if (_pop.size() - _archive.size()>0){
      resizeValue=_pop.size() - _archive.size();
      _pop.shuffle();      
    }
    else
      resizeValue=0;
    
    for(unsigned int i=0; i<_archive.size(); i++)
      tab.push_back(i);
    std::random_shuffle(tab.begin(), tab.end(), rndGen);
    
    _pop.resize(resizeValue);
    
    for (unsigned int i=0; i<(popCorrectSize-resizeValue); i++)
      {
    	_pop.push_back(_archive[tab[i]]);
      }
    for (unsigned int i=0; i<_pop.size(); i++)
      {
    	_pop[i].fitness(0.0);
    	_pop[i].diversity(0.0);
      }
  }
  };*/



/*
 *
 */
template <class EOT>
class IBEAAggregation: public ArchiveToPopAggregation<EOT>{
public:
  void operator()(eoPop< EOT >& _pop, moeoArchive< EOT >& _archive) {
    unsigned int popSize= _pop.size();
    _pop.reserve(popSize + _archive.size());
    for (unsigned int i=0; i<_archive.size(); i++)
      {
	_pop.push_back(_archive[i]);
    	_pop[i+popSize].fitness(0.0);
    	//_pop[i].diversity(0.0);
      }
    diversityAssignment(_pop);
    std::sort(_pop.begin(), _pop.end(), cmp);
    _pop.resize(popSize);
  }

private:
  moeoCrowdingDiversityAssignment<EOT> diversityAssignment;
  moeoDiversityThenFitnessComparator<EOT> cmp;
};


/*
 *
 */
/*template <class EOT>
class IBEAAggregationQuiMarchePas{
public:
  void operator()(eoPop< EOT >& _pop, moeoArchive< EOT >& _archive) {
    UF_random_generator<unsigned int> rndGen;    
    std::vector<unsigned int> tab;
    unsigned int resizeValue;
    unsigned int popCorrectSize= _pop.size();


    if (_pop.size() <= _archive.size())
      _pop.resize(0);
    
    for (unsigned int i=0; i<_archive.size(); i++)
      {
	_pop.push_back(_archive[i]);
    	_pop[i].fitness(0.0);
    	_pop[i].diversity(0.0);
      }
    
  }
  };*/


/*
 *
 */
template <class EOT>
class LSAggregation: public ArchiveToArchiveAggregation<EOT>{
public:

 LSAggregation(eoMonOp <EOT> & _op, eoEvalFunc<EOT>& _eval, unsigned int _nbKick): ArchiveToArchiveAggregation<EOT>(),op(_op), eval(_eval), nbKick(_nbKick){}

  void operator()(moeoArchive< EOT >& _archive1, moeoArchive <EOT >& _archive2) {
    unsigned int archSize=_archive2.size();
    if(archSize>0){
      _archive1.resize(0);
      _archive1.push_back(_archive2[rng.random(archSize)]);
      //      std::cout << "kick : " << nbKick << std::endl;
      //si la solution n'a pas encore été visité
      if(_archive1[0].flag()==1){
	std::cout << "kick pour du vrai" << std::endl;
	//on la kick
	for(unsigned int i=0; i<nbKick; i++){
	  //std::cout << _archive1[0] << std::endl;
	  op(_archive1[0]);
	}
	eval(_archive1[0]);
	_archive1[0].flag(0);
      }
      for (unsigned int i=0; i<_archive1.size(); i++)
	{

	  _archive1[i].fitness(0.0);
	  _archive1[i].diversity(0.0);
      }
    }
  }

private:
  eoMonOp<EOT> & op;
  eoEvalFunc<EOT> & eval;
  unsigned int nbKick;

};


//Aggregation pour archiver une pop
template <class EOT>
class CentralAggregation: public PopToArchiveAggregation<EOT>{
public:
  void operator()(moeoArchive< EOT >& _archive, eoPop< EOT >& _pop) {
    _archive(_pop);
  }
};

//aggregation pour archiver une archive
template <class EOT>
class ArchToArchAggregation{
public:
  void operator()(moeoArchive< EOT >& _archive1, moeoArchive< EOT >& _archive2) {
    _archive1(_archive2);
  }
};

template <class EOT>
class gestArchive{

 public:
  gestArchive(moeoArchive <EOT>& _archive,
		 eoPop <EOT>& _pop,
		 peoSyncDataTransfer & _syncDataTransfer,
		 eoContinue <EOT>& _cont):
    archive(_archive), pop(_pop), syncDataTransfer(_syncDataTransfer), cont(_cont){}

  void operator()(eoPop <EOT>& population) {
    while(true){
      //      pop.resize(0);
      //for(unsigned int i=0; i<archive.size(); i++)
      //	pop.push_back(archive[i]);
      syncDataTransfer();//appel pour faire l'echange
      //      archive(pop);
      //      pop.resize(0);
      //      for(unsigned int i=0; i<archive.size(); i++)
      //	pop.push_back(archive[i]);
      cont(archive);
    }
  }

 private:
  moeoArchive <EOT>& archive;
  eoPop <EOT>& pop;
  peoSyncDataTransfer& syncDataTransfer;
  eoContinue <EOT>& cont;
};


template <class EOT>
class testPEO{

 public:
  testPEO(int _argc,
	  char** _argv,
	  moeoPopAlgo<EOT> & _algo1,
	  moeoPopAlgo<EOT> & _algo2,
	  eoPop<EOT> & _pop1,
	  eoPop<EOT> & _pop2,
	  moeoNewBoundedArchive<EOT> & _centralArchive,
	  AbstractEntityToArchiveAggregation<EOT, eoPop<EOT> &, void>& _algo1ToArchive,
	  AbstractEntityToArchiveAggregation<EOT, eoPop<EOT> &, void>& _algo2ToArchive,
	  generalAggregation <eoPop<EOT> &, moeoArchive<EOT> &, void>& _archiveToAlgo1,
	  generalAggregation <moeoArchive<EOT> &, moeoArchive<EOT> &, void>& _archiveToAlgo2,
	  eoCheckPoint<EOT> & _checkAlgo1,
	  eoCheckPoint<EOT> & _checkAlgo2,
	  eoCheckPoint<EOT> & _checkArchive):
  argc(_argc),
    argv(_argv),
    algo1(_algo1),
    algo2(_algo2),
    pop1(_pop1),
    pop2(_pop2),
    centralArchive(_centralArchive),
    algo1ToArchive(_algo1ToArchive),
    algo2ToArchive(_algo2ToArchive),
    archiveToAlgo1(_archiveToAlgo1),
    archiveToAlgo2(_archiveToAlgo2),
    checkAlgo1(_checkAlgo1),
    checkAlgo2(_checkAlgo2),
    checkArchive(_checkArchive){}

  void operator()(){

    //PEO Initialization 
    peo :: init (argc, argv);

    //Two RingTopolgy
    RingTopology ring1, ring2;


    //DataTransfer for the fisrt ring
    peoSyncDataTransfer transfer1(centralArchive, ring1, algo1ToArchive);
    peoSyncDataTransfer transfer2(pop1, ring1, archiveToAlgo1);

    //DataTransfer for the second ring
    peoSyncDataTransfer transfer3(centralArchive, ring2, algo2ToArchive);
    peoSyncDataTransfer transfer4(pop2, ring2, archiveToAlgo2);

    //Transfer Algo1 -> archiveCentral (Ring1)
    gestArchive<EOT> toCenter1(centralArchive, pop1, transfer1, checkArchive);

    //Transfer archiveCentral -> Algo1 (Ring1)
    eoGenContinue <EOT> genContinuator1(100);
    gestionTransfer<EOT> exitCenter1(genContinuator1, transfer2, centralArchive);
    checkAlgo1.add(exitCenter1);

    //Transfer Algo2 -> archiveCentral (Ring2)
    gestArchive<EOT> toCenter2(centralArchive, pop2, transfer3, checkArchive);

    //Transfer archiveCentral -> Algo2 (Ring2)
    eoGenContinue <EOT> genContinuator2(200);
    gestionTransfer<EOT> exitCenter2(genContinuator2, transfer4, centralArchive);
    checkAlgo2.add(exitCenter2);

    //dummyPop
    eoPop<EOT> dummyPop;

    //Wrapping of algotithm
    peoParallelAlgorithmWrapper parallelEA_1(toCenter1, dummyPop);
    transfer1.setOwner( parallelEA_1 );
    
    
    peoParallelAlgorithmWrapper parallelEA_2(algo1, pop1);
    transfer2.setOwner( parallelEA_2 );
    
    
    peoParallelAlgorithmWrapper parallelEA_3(toCenter2, dummyPop);
    transfer3.setOwner( parallelEA_3 );
    
    
    peoParallelAlgorithmWrapper parallelEA_4(algo2, pop2);
    transfer4.setOwner( parallelEA_4 );
    
    
    //run 
    peo :: run( );
    peo :: finalize( );
    endDebugging();
  }

 private:
  int argc;
  char** argv;
  moeoPopAlgo<EOT> & algo1;
  moeoPopAlgo<EOT> & algo2;
  eoPop<EOT> & pop1;
  eoPop<EOT> & pop2;
  moeoNewBoundedArchive <EOT> & centralArchive;
  AbstractEntityToArchiveAggregation<EOT, eoPop<EOT> &, void> & algo1ToArchive;
  AbstractEntityToArchiveAggregation<EOT, eoPop<EOT> &, void> & algo2ToArchive;
  generalAggregation <eoPop<EOT> &, moeoArchive<EOT> &, void> & archiveToAlgo1;
  generalAggregation <moeoArchive<EOT> &, moeoArchive<EOT> &, void> & archiveToAlgo2;
  eoCheckPoint<EOT> & checkAlgo1;
  eoCheckPoint<EOT> & checkAlgo2;
  eoCheckPoint<EOT> & checkArchive;
  

};
