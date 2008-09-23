/*
* <FlowShopBenchmarkParser.h>
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

#ifndef FLOWSHOPBENCHMARKPARSER_H_
#define FLOWSHOPBENCHMARKPARSER_H_

#include <stdlib.h>
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
