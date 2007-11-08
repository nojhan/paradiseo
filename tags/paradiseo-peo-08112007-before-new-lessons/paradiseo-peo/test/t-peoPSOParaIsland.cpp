/* 
* <t-peoPSOParaIsland.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
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
* with loading,  using,  modifying and/or developing or reproducing the
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
*          clive.canape@inria.fr
*/
//-----------------------------------------------------------------------------
// t-peoPSOParaIsland.cpp
//-----------------------------------------------------------------------------

#include <peo>

typedef eoRealParticle < double >Indi;


//Evaluation function
double f (const Indi & _indi)
{  

    const double PI = 4.0 * atan(1.0);
    double sum=0.;
    for (unsigned i = 0; i < _indi.size()-1; i++)
    	sum+=pow((_indi[i]-1),2)*(1+pow(sin(3*PI*_indi[i+1]),2));
    sum+=pow(sin(3*PI*_indi[0]),2);
    sum+=(_indi[_indi.size()-1]-1)*(1+pow(sin(2*PI*_indi[_indi.size()-1]),2));
    return (-sum);
    
}

template < class POT > class eoPrint : public eoContinue <POT> 
{
	public :
	
		 	bool operator () (const eoPop <POT> & __pop) 
			{
				double result[__pop.size()];
				for(unsigned i=0;i<__pop.size();i++)
					result[i]=__pop[i].best();
				std::sort(result,result+__pop.size());
				std::cout << "\n"<<result[__pop.size()-1];
				return true ;
			}
} ;


int main (int __argc, char *__argv[])
{

//Initialization
  peo :: init( __argc, __argv );
//Parameters
 
	const unsigned int  MIG_FREQ = 1; 
    const unsigned int VEC_SIZE = 4;        
    const unsigned int POP_SIZE = 10;        
    const unsigned int NEIGHBORHOOD_SIZE= 5; 
    const unsigned int MAX_GEN = 100;
    const double INIT_POSITION_MIN = -50.0;
    const double INIT_POSITION_MAX = 50.0;
    const double INIT_VELOCITY_MIN = -1;
    const double INIT_VELOCITY_MAX = 1;
    const double C1 = 0.5; 
    const double C2 = 2;
    rng.reseed (44);
    if(getNodeRank()==1)
		std::cout<<"\n\nWith two islands\n\n";
	peoEvalFuncPSO<Indi, double, const Indi& > plainEval(f);
    eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
	eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
    eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
    eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
    eoFirstIsBestInit < Indi > localInit;
    eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
    eoStandardFlight < Indi > flight(bndsFlight); 
    eoEvalFuncCounter < Indi > evalSeq (plainEval); 
    peoParaPopEval< Indi > eval(plainEval);  
    eoPop < Indi > pop;
    pop.append (POP_SIZE, random);    
    apply(evalSeq, pop);
    apply < Indi > (veloRandom, pop);
    apply < Indi > (localInit, pop);
    eoLinearTopology<Indi> topology(NEIGHBORHOOD_SIZE);
    topology.setup(pop);
    eoRealVectorBounds bnds(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
    eoStandardVelocity < Indi > velocity (topology,C1,C2,bnds);
    eoGenContinue < Indi > genContPara (MAX_GEN);
    eoPrint <Indi>  print;
	eoCombinedContinue <Indi> continuator(genContPara);
	continuator.add(print);
    eoCheckPoint<Indi> checkpoint(continuator); 
    RingTopology topologyMig;
 /*******************************************************************/  
    eoPeriodicContinue< Indi > mig_cont( MIG_FREQ );
    peoPSOSelect<Indi> mig_selec(topology); 
	eoSelectNumber< Indi > mig_select(mig_selec);
	peoPSOReplacement<Indi> mig_replace;
/*****************************************************************/	 
    
    peoEvalFuncPSO<Indi, double, const Indi& > plainEval2(f);
    eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
	eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen);
    eoUniformGenerator < double >sGen2 (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
    eoVelocityInitFixedLength < Indi > veloRandom2 (VEC_SIZE, sGen2);
    eoFirstIsBestInit < Indi > localInit2;
    eoRealVectorBounds bndsFlight2(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
    eoStandardFlight < Indi > flight2(bndsFlight2); 
    eoEvalFuncCounter < Indi > evalSeq2 (plainEval2); 
    peoParaPopEval< Indi > eval2(plainEval2);  
    eoPop < Indi > pop2;
    pop2.append (POP_SIZE, random2);    
    apply(evalSeq2, pop2);
    apply < Indi > (veloRandom2, pop2);
    apply < Indi > (localInit2, pop2);
    eoLinearTopology<Indi> topology2(NEIGHBORHOOD_SIZE);
    topology2.setup(pop2);
    eoRealVectorBounds bnds2(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
    eoStandardVelocity < Indi > velocity2 (topology2,C1,C2,bnds2);
    eoGenContinue < Indi > genContPara2 (MAX_GEN);
    eoCheckPoint<Indi> checkpoint2(genContPara2); 
 /*******************************************************************/  
    eoPeriodicContinue< Indi > mig_cont2( MIG_FREQ );
    peoPSOSelect<Indi> mig_selec2(topology2); 
	eoSelectNumber< Indi > mig_select2(mig_selec2);
	peoPSOReplacement<Indi> mig_replace2;

	peoAsyncIslandMig< Indi > mig( mig_cont, mig_select, mig_replace, topologyMig, pop, pop2);
	checkpoint.add( mig );
	peoAsyncIslandMig< Indi > mig2( mig_cont2, mig_select2, mig_replace2, topologyMig, pop2, pop);
	checkpoint2.add( mig2 );
	
    peoPSO < Indi > psa(checkpoint, eval, velocity, flight);
    mig.setOwner( psa );
    psa(pop);  
    peoPSO < Indi > psa2(checkpoint2, eval2, velocity2, flight2);
   	mig2.setOwner( psa2 );
    psa2(pop2);
  
    peo :: run(); 
    peo :: finalize();
}

