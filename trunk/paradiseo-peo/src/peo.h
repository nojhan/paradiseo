/*
* <peo.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
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
*
*/

#ifndef __peo_h_
#define __peo_h_

#include <eo>
#include <oldmo>
#include <moeo>

/** @mainpage Welcome to Paradiseo-PEO

    @section Introduction

    PEO is an extension of the ANSI-C++ compliant evolutionary computation library EO.
    <BR>
    It contains classes for the most common parallel and distributed models and hybridization mechanisms.

    @section authors AUTHORS

    <TABLE>
    <TR>
      <TD>
        Sebastien CAHON
      </TD>
    </TR>
    <TR>
      <TD>
        Alexandru-Adrian TANTAR
      </TD>
    </TR>
    <TR>
      <TD>
        Clive Canape
      </TD>
    </TR>
    </TABLE>

    @section LICENSE

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


    @section Paradiseo Home Page

    <A href=http://paradiseo.gforge.inria.fr>http://paradiseo.gforge.inria.fr</A>

    @section Installation

    The installation procedure of the package is detailed in the
    <a href="../../README">README</a> file in the top-directory of the source-tree.

*/

#include "core/peo_init.h"
#include "core/peo_run.h"
#include "core/peo_fin.h"

#include "core/messaging.h"
#include "core/eoPop_mesg.h"
#include "core/eoVector_mesg.h"

#include "peoWrapper.h"

/* <------- components for parallel algorithms -------> */
#include "peoTransform.h"
#include "peoEvalFunc.h"
#include "peoPopEval.h"
#include "peoMoeoPopEval.h"

/* Cooperative island model */
#include "core/ring_topo.h"
#include "core/star_topo.h"
#include "core/random_topo.h"
#include "core/complete_topo.h"
#include "peoData.h"
#include "peoSyncIslandMig.h"
#include "peoAsyncIslandMig.h"
#include "peoAsyncDataTransfer.h"
#include "peoSyncDataTransfer.h"

/* Synchronous multi-start model */
#include "peoMultiStart.h"
/* <------- components for parallel algorithms -------> */

/* Parallel PSO */
#include "peoPSO.h"

#endif
