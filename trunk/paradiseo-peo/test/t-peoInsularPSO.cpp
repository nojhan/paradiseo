/* 
* <t-peoInsularPSO.cpp>
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
// t-peoInsularPSO.cpp
//-----------------------------------------------------------------------------

#include <peo>

#define N 4 

typedef eoRealParticle < double >Indi;
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

void peoPSOSeq ()
{
	
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
    std::cout<<"\n\nWith one PSO\n\n";
 	eoEvalFuncPtr<Indi, double, const Indi& > plainEvalSeq(f);
    eoEvalFuncCounter < Indi > evalSeq (plainEvalSeq);                
    eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
    eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
    eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
    eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
    eoFirstIsBestInit < Indi > localInit;
    eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
    eoStandardFlight < Indi > flight(bndsFlight);    
    eoPop < Indi > popSeq;
    popSeq.append (POP_SIZE, random);  
    apply(evalSeq, popSeq);;
    apply < Indi > (veloRandom, popSeq);
    apply < Indi > (localInit, popSeq);
    eoLinearTopology<Indi> topologySeq(NEIGHBORHOOD_SIZE);
    topologySeq.setup(popSeq);
    eoRealVectorBounds bndsSeq(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
    eoStandardVelocity < Indi > velocitySeq (topologySeq,C1,C2,bndsSeq);
    eoGenContinue < Indi > genContSeq (MAX_GEN);
    eoPrint <Indi>  printSeq;
	eoCombinedContinue <Indi> continuatorSeq(genContSeq);
	continuatorSeq.add(printSeq);
    eoCheckPoint<Indi> checkpointSeq(continuatorSeq);   
    eoSyncEasyPSO < Indi > psaSeq(checkpointSeq, evalSeq, velocitySeq, flight);
    psaSeq (popSeq); 
    popSeq.sort (); 
}

void peoPSOPara()
{ 
	char *tmp="mpiexec -n ",*tmp2=" ./t-peoPSOParaIsland @lesson.param ",tmp3[4],buffer[256];
	sprintf(tmp3,"%d",N);
	strcpy(buffer,tmp);
	strcat(buffer,tmp3);
	strcat(buffer,tmp2);
	system(buffer);
}

int main (int __argc, char *__argv[])
{	

	peoPSOSeq ();
	peoPSOPara();
}
