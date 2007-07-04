// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopBenchmarkParser.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPBENCHMARKPARSER_H_
#define FLOWSHOPBENCHMARKPARSER_H_

#include <fstream>
#include <string>
#include <vector>

/**
 * Class to handle parameters of a flow-shop instance from a benchmark file
 */
class FlowShopBenchmarkParser
{
public:

    /**
     * Ctor
     * @param _benchmarkFileName the name of the benchmark file
     */
    FlowShopBenchmarkParser(const std::string _benchmarkFileName);


    /**
     * the number of machines
     */
    const unsigned int getM();


    /**
     * the number of jobs
     */
    const unsigned int getN();


    /**
     * the processing times
     */
    const std::vector < std::vector < unsigned int > > getP();


    /**
     * the due-dates
     */
    const std::vector < unsigned int > getD();


    /**
     * printing...
     */
    void printOn(std::ostream & _os) const;


private:

    /** number of machines */
    unsigned int M;
    /** number of jobs */
    unsigned int N;
    /** p[i][j] = processing time of job j on machine i */
    std::vector < std::vector < unsigned int > > p;
    /** d[j] = due-date of the job j */
    std::vector < unsigned int > d;


    /**
     * Initialisation of the parameters with the data contained in the benchmark file
     * @param _benchmarkFileName the name of the benchmark file
     */
    void init(const std::string _benchmarkFileName);

};

#endif /*FLOWSHOPBENCHMARKPARSER_H_*/
