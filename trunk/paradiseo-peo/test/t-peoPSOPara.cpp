/*
* <t-peoPSOPara.cpp>
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
// t-peoPSOPara.cpp
//-----------------------------------------------------------------------------

#include <peo>
#include "../src/rmc/mpi/schema.h"
#include "../src/rmc/mpi/node.h"

typedef eoRealParticle < double >Indi;
double f (const Indi & _indi)
{
  double sum = 0;
  for (unsigned i = 0; i < _indi.size (); i++)
    sum += pow(_indi[i],2);
  return (-sum);
}

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  clock_t	begin,end;
  double time;
  const unsigned int VEC_SIZE = 2;
  const unsigned int POP_SIZE = 10;
  const unsigned int NEIGHBORHOOD_SIZE= 5;
  const unsigned int MAX_GEN = 500;
  const double FIT_CONT = -1e-6;
  const double INIT_POSITION_MIN = -5.0;
  const double INIT_POSITION_MAX = 5.0;
  const double INIT_VELOCITY_MIN = -1;
  const double INIT_VELOCITY_MAX = 1;
  const double C1 = 2;
  const double C2 = 2;
  rng.reseed (36);
  if ((isScheduleNode())&&(__argc >= 3))
    begin=clock();
  peoEvalFuncPSO<Indi, double, const Indi& > plainEvalPara(f);
  eoEvalFuncCounter < Indi > evalStartPara (plainEvalPara);
  peoParaPopEval< Indi > evalPara(plainEvalPara);
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoPop < Indi > popPara;
  popPara.append (POP_SIZE, random);
  apply(evalStartPara, popPara);
  apply < Indi > (veloRandom, popPara);
  apply < Indi > (localInit, popPara);
  eoLinearTopology<Indi> topologyPara(NEIGHBORHOOD_SIZE);
  topologyPara.setup(popPara);
  eoRealVectorBounds bndsPara(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocityPara (topologyPara,C1,C2,bndsPara);
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoFitContinue < Indi > fitContPara (FIT_CONT);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  continuatorPara.add(fitContPara);
  eoCheckPoint<Indi> checkpointPara(continuatorPara);
  peoPSO < Indi > psaPara(checkpointPara, evalPara, velocityPara, flight);
  psaPara(popPara);
  peo :: run();
  peo :: finalize();
  if ((isScheduleNode())&&(__argc >= 3))
    {
      end=clock();
      time=end-begin;
      std::cout<<"\n\nEfficiency :  "<<(atoi(__argv[2])/(time*getNumberOfNodes()))<<"\n";
    }
  return 1;
}
