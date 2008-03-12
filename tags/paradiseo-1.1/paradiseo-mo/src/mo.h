/*
  <mo.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
  (C) OPAC Team, LIFL, 2002-2007

  Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef _mo_h
#define _mo_h

#include <moAlgo.h>
#include <moAspirCrit.h>
#include <moBestImprSelect.h>
#include <moComparator.h>
#include <moCoolingSchedule.h>
#include <moExponentialCoolingSchedule.h>
#include <moFirstImprSelect.h>
#include <moFitComparator.h>
#include <moFitSolContinue.h>
#include <moGenSolContinue.h>
#include <moHC.h>
#include <moHCMoveLoopExpl.h>
#include <moILS.h>
#include <moImprBestFitAspirCrit.h>
#include <moItRandNextMove.h>
#include <moLinearCoolingSchedule.h>
#include <moLSCheckPoint.h>
#include <moMoveExpl.h>
#include <moMove.h>
#include <moMoveIncrEval.h>
#include <moMoveInit.h>
#include <moMoveLoopExpl.h>
#include <moMoveSelect.h>
#include <moNextMove.h>
#include <moNoAspirCrit.h>
#include <moNoFitImprSolContinue.h>
#include <moRandImprSelect.h>
#include <moRandMove.h>
#include <moSA.h>
#include <moSimpleMoveTabuList.h>
#include <moSimpleSolutionTabuList.h>
#include <moSolContinue.h>
#include <moSteadyFitSolContinue.h>
#include <moTabuList.h>
#include <moTS.h>
#include <moTSMoveLoopExpl.h>

#endif
