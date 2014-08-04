/*
* <FlowShopEval.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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
//-----------------------------------------------------------------------------

#include "FlowShopEval.h"


FlowShopEval::FlowShopEval(unsigned int _M, unsigned int _N, const std::vector< std::vector<unsigned int> > & _p, const std::vector<unsigned int> & _d) :
        M(_M), N (_N), p(_p), d(_d)
{}


void FlowShopEval::operator()(FlowShop & _flowshop)
{
    FlowShopObjectiveVector objVector;
    objVector[0] = makespan(_flowshop);
    objVector[1] = tardiness(_flowshop);
    _flowshop.objectiveVector(objVector);
}



double FlowShopEval::makespan(const FlowShop & _flowshop)
{
    // completion times computation for each job on each machine
    // C[i][j] = completion of the jth job of the scheduling on the ith machine
    std::vector< std::vector<unsigned int> > C = completionTime(_flowshop);
    return C[M-1][_flowshop[N-1]];
}


double FlowShopEval::tardiness(const FlowShop & _flowshop)
{
    // completion times computation for each job on each machine
    // C[i][j] = completion of the jth job of the scheduling on the ith machine
    std::vector< std::vector<unsigned int> > C = completionTime(_flowshop);
    // tardiness computation
    unsigned int long sum = 0;
    for (unsigned int j=0 ; j<N ; j++)
        sum += (unsigned int) std::max (0, (int) (C[M-1][_flowshop[j]] - d[_flowshop[j]]));
    return sum;
}


std::vector< std::vector<unsigned int> > FlowShopEval::completionTime(const FlowShop & _flowshop)
{
    std::vector< std::vector<unsigned int> > C;
    C.resize(M);
    for (unsigned int i=0;i<M;i++)
    {
        C[i].resize(N);
    }
    C[0][_flowshop[0]] = p[0][_flowshop[0]];
    for (unsigned int j=1; j<N; j++)
        C[0][_flowshop[j]] = C[0][_flowshop[j-1]] + p[0][_flowshop[j]];
    for (unsigned int i=1; i<M; i++)
        C[i][_flowshop[0]] = C[i-1][_flowshop[0]] + p[i][_flowshop[0]];
    for (unsigned int i=1; i<M; i++)
        for (unsigned int j=1; j<N; j++)
            C[i][_flowshop[j]] = std::max(C[i][_flowshop[j-1]], C[i-1][_flowshop[j]]) + p[i][_flowshop[j]];
    return C;
}
