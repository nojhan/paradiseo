// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "FlowShopBenchmarkParser.h"

// (c) OPAC Team, LIFL, March 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/

#ifndef _FlowShopBenchmarkParser_h
#define _FlowShopBenchmarkParser_h

// general include
#include <stdexcept>
#include <fstream>

/** Web site to download benchmarks  */
const static
  std::string
  BENCHMARKS_WEB_SITE = "www.lifl.fr/~liefooga/benchmarks/";


/** 
 * Class to handle parameters of a flow-shop instance from a benchmark file
 * benchmark files are available at www.lifl.fr/~basseur/BenchsUncertain/
 */
class FlowShopBenchmarkParser
{

public:

  /**
   * constructor
   * @param const string _benchmarkFileName  the name of the benchmark file
   */
  FlowShopBenchmarkParser (const string _benchmarkFileName)
  {
    init (_benchmarkFileName);
  }

  /**
   * the number of machines
   */
  const unsigned
  getM ()
  {
    return M;
  }

  /**
   * the number of jobs
   */
  const unsigned
  getN ()
  {
    return N;
  }

  /**
   * the processing times
   */
  const
    std::vector < std::vector < unsigned > >
  getP ()
  {
    return p;
  }

  /**
   * the due-dates
   */
  const
  std::vector < unsigned >
  getD ()
  {
    return d;
  }

  /** 
   * printing...
   */
  void
  printOn (ostream & _os) const
  {
    _os <<
      "M=" <<
      M <<
      " N=" <<
      N <<
      endl;
    _os <<
      "*** processing times" <<
      endl;
    for (unsigned i = 0; i < M; i++)
      {
	for (unsigned j = 0; j < N; j++)
	  {
	    _os << p[i][j] << " ";
	  }
	_os <<
	  endl;
      }
    _os << "*** due-dates" << endl;
    for (unsigned j = 0; j < N; j++)
      {
	_os << d[j] << " ";
      }
    _os << endl << endl;
  }

private:
  /** number of machines */
  unsigned
    M;
  /** number of jobs */
  unsigned
    N;
  /** p[i][j] = processing time of job j on machine i */
  std::vector < std::vector < unsigned > >
    p;
  /** d[j] = due-date of the job j */
  std::vector < unsigned >
    d;


  /**
   * Initialisation of the parameters with the data contained in the benchmark file
   * @param const string _benchmarkFileName  the name of the benchmark file
   */
  void
  init (const string _benchmarkFileName)
  {
    string buffer;
    string::size_type start, end;
    ifstream inputFile (_benchmarkFileName.data (), ios::in);
    // opening of the benchmark file
    if (!inputFile)
      cerr << "*** ERROR : Unable to open the benchmark file '" <<
	_benchmarkFileName << "'" << endl;
    // number of jobs (N)
    getline (inputFile, buffer, '\n');
    N = atoi (buffer.data ());
    // number of machines M
    getline (inputFile, buffer, '\n');
    M = atoi (buffer.data ());
    // initial and current seeds (not used)
    getline (inputFile, buffer, '\n');
    // processing times and due-dates
    p = std::vector < std::vector < unsigned > > (M, N);
    d = std::vector < unsigned > (N);
    // for each job...
    for (unsigned j = 0; j < N; j++)
      {
	// index of the job (<=> j)
	getline (inputFile, buffer, '\n');
	// due-date of the job j
	getline (inputFile, buffer, '\n');
	d[j] = atoi (buffer.data ());
	// processing times of the job j on each machine
	getline (inputFile, buffer, '\n');
	start = buffer.find_first_not_of (" ");
	for (unsigned i = 0; i < M; i++)
	  {
	    end = buffer.find_first_of (" ", start);
	    p[i][j] = atoi (buffer.substr (start, end - start).data ());
	    start = buffer.find_first_not_of (" ", end);
	  }
      }
    // closing of the input file
    inputFile.close ();
  }

};

#endif
