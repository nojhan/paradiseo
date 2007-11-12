/* 
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
*
* Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/syncor developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*          
*/

#include <peo>
#include <es.h>

typedef eoReal<double> Indi;

double f (const Indi & _indi)
{  
    double sum;
    sum=_indi[1]-pow(_indi[0],2);
    sum=100*pow(sum,2);
    sum+=pow((1-_indi[0]),2);
    return (-sum);
}

int main (int __argc, char *__argv[])
{

    peo :: init( __argc, __argv );
    const unsigned int VEC_SIZE = 2;       
    const unsigned int POP_SIZE = 20; 
    const unsigned int MAX_GEN = 300; 
    const double INIT_POSITION_MIN = -2.0;  
    const double INIT_POSITION_MAX = 2.0;  
    const float CROSS_RATE = 0.8; 
    const double EPSILON = 0.01;  
    const float MUT_RATE = 0.3; 
 // MIG_FREQ define the frequency of the migration. 
    const unsigned int  MIG_FREQ = 10;
 // MIG_SIZE define the size of each migration.
    const unsigned int  MIG_SIZE = 5;
    rng.reseed (time(0));
    
 // Define the topology of your island model
 	RingTopology topology;
 	
 // First algorithm   
 /*****************************************************************************************/     
    eoGenContinue < Indi > genContPara (MAX_GEN);
    eoCombinedContinue <Indi> continuatorPara (genContPara);
    eoCheckPoint<Indi> checkpoint(continuatorPara);  
  	peoEvalFunc<Indi> plainEval(f);
	peoSeqPopEval< Indi > eval(plainEval);  // Here, the evaluation is sequential
    eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
    eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
	eoRankingSelect<Indi> selectionStrategy;
	eoSelectNumber<Indi> select(selectionStrategy,POP_SIZE);
    eoSegmentCrossover<Indi> crossover;
    eoUniformMutation<Indi>  mutation(EPSILON); 
    eoSGATransform<Indi> transform(crossover,CROSS_RATE,mutation,MUT_RATE);
	peoSeqTransform<Indi> eaTransform(transform); // Here, the transformation is sequential
	eoPlusReplacement<Indi> replace;  
    eoPop < Indi > pop;
    pop.append (POP_SIZE, random);    
    
    eoPeriodicContinue <Indi> mig_cont( MIG_FREQ );  // Migration occurs periodically
	eoRandomSelect<Indi> mig_select_one; 		// Emigrants are randomly selected 
	eoSelectNumber<Indi> mig_select (mig_select_one,MIG_SIZE);
	eoPlusReplacement<Indi> mig_replace; 		// Immigrants replace the worse individuals

/*****************************************************************************************/     

// Second algorithm (on the same model but with others names)  
 /*****************************************************************************************/     
    eoGenContinue < Indi > genContPara2 (MAX_GEN);
    eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
    eoCheckPoint<Indi> checkpoint2(continuatorPara2);  
  	peoEvalFunc<Indi> plainEval2(f);
	peoSeqPopEval< Indi > eval2(plainEval2);
    eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
    eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
	eoRankingSelect<Indi> selectionStrategy2;
	eoSelectNumber<Indi> select2(selectionStrategy2,POP_SIZE);
    eoSegmentCrossover<Indi> crossover2;
    eoUniformMutation<Indi>  mutation2(EPSILON); 
    eoSGATransform<Indi> transform2(crossover2,CROSS_RATE,mutation2,MUT_RATE);
	peoSeqTransform<Indi> eaTransform2(transform2);
	eoPlusReplacement<Indi> replace2;  
    eoPop < Indi > pop2;
    pop2.append (POP_SIZE, random2);    
    eoPeriodicContinue <Indi> mig_cont2( MIG_FREQ ); 
	eoRandomSelect<Indi> mig_select_one2; 		
	eoSelectNumber<Indi> mig_select2 (mig_select_one2,MIG_SIZE);
	eoPlusReplacement<Indi> mig_replace2; 		

/*****************************************************************************************/     

// You can choose between :
//
//      - Synchronous communication : peoSyncIslandMig<Indi> mig(MIG_FREQ,mig_select,mig_replace,topology,pop,pop);
//      - Asynchronous communication : peoAsyncIslandMig<Indi> mig(mig_cont,mig_select,mig_replace,topology,pop,pop);
// With a grid, you should use an asynchronous communication
	peoAsyncIslandMig<Indi> mig(mig_cont,mig_select,mig_replace,topology,pop,pop2);
	checkpoint.add(mig);
	peoAsyncIslandMig<Indi> mig2(mig_cont2,mig_select2,mig_replace2,topology,pop2,pop);
	checkpoint2.add(mig2);
	
// Initialization of the algorithms 
 	peoEA<Indi> Algo(checkpoint,eval,select,eaTransform,replace);
	mig.setOwner(Algo);
	Algo(pop);	
	peoEA<Indi> Algo2(checkpoint2,eval2,select2,eaTransform2,replace2);
	mig2.setOwner(Algo2);
	Algo2(pop2);
	
    peo :: run(); 
    peo :: finalize();
    if(getNodeRank()==1)
    {
		std::cout << "Final population 1 :\n" << pop << std::endl;
		std::cout << "Final population 2 :\n" << pop2 << std::endl;
    }
}
