// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopEval.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <FlowShopEval.h>


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


std::vector< std::vector<unsigned int> > FlowShopEval::completionTime(const FlowShop & _flowshop) {
    std::vector< std::vector<unsigned int> > C(M,N);
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
