// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopEval.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPEVAL_H_
#define FLOWSHOPEVAL_H_

#include <vector>
#include <core/moeoEvalFunc.h>
#include <FlowShop.h>

/**
 * Evaluation of the objective vector a (multi-objective) FlowShop object
 */
class FlowShopEval : public moeoEvalFunc<FlowShop>
{
public:

    /**
     * Ctor
     * @param _M the number of machines 
     * @param _N the number of jobs to schedule
     * @param _p the processing times
     * @param _d the due dates
     */
    FlowShopEval(unsigned int _M, unsigned int _N, const std::vector< std::vector<unsigned int> > & _p, const std::vector<unsigned int> & _d);


    /**
     * computation of the multi-objective evaluation of a FlowShop object
     * @param _flowshop the FlowShop object to evaluate
     */
    void operator()(FlowShop & _flowshop);


private:

    /** number of machines */
    unsigned int M;
    /** number of jobs */
    unsigned int N;
    /** p[i][j] = processing time of job j on machine i */
    std::vector< std::vector < unsigned int > > p;
    /** d[j] = due-date of the job j */
    std::vector < unsigned int > d;


    /**
     * computation of the makespan
     * @param _flowshop the genotype to evaluate
     */
    double makespan(const FlowShop & _flowshop);


    /**
     * computation of the tardiness
     * @param _flowshop the genotype to evaluate
     */
    double tardiness(const FlowShop & _flowshop);


    /**
     * computation of the completion times of a scheduling (for each job on each machine)
     * C[i][j] = completion of the jth job of the scheduling on the ith machine
     * @param _flowshop the genotype to evaluate
     */
    std::vector< std::vector<unsigned int> > completionTime (const FlowShop & _flowshop);

};

#endif /*FLOWSHOPEVAL_H_*/
