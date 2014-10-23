/*
  <moSampling.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

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

#ifndef moSampling_h
#define moSampling_h

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <eoFunctor.h>
#include <utils/eoMonitor.h>
#include <continuator/moStat.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moVectorMonitor.h>
#include <algo/moLocalSearch.h>
#include <eoInit.h>


/**
 * To sample the search space:
 * A local search is used to sample the search space
 * Some statistics are computed at each step of the local search
 *
 * Can be used to study the fitness landscape
 */
template <class Neighbor>
class moSampling : public eoF<void>
{
public:
	typedef typename Neighbor::EOT EOT ;

	/**
	 * Constructor
	 * @param _init initialisation method of the solution
	 * @param _localSearch  local search to sample the search space
	 * @param _stat statistic to compute during the search
	 * @param _monitoring the statistic is saved into the monitor if true
	 */
	template <class ValueType>
	moSampling(eoInit<EOT> & _init, moLocalSearch<Neighbor> & _localSearch, moStat<EOT,ValueType> & _stat, bool _monitoring = true) : init(_init), localSearch(&_localSearch), continuator(_localSearch.getContinuator())
	{
		checkpoint = new moCheckpoint<Neighbor>(*continuator);
		add(_stat, _monitoring);
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * default destructor
	 */
	~moSampling() {
		// delete all monitors
		for (unsigned i = 0; i < monitorVec.size(); i++)
			delete monitorVec[i];

		// delete the checkpoint
		delete checkpoint ;
	}

	/**
	 * Add a statistic
	 * @param _stat another statistic to compute during the search
	 * @param _monitoring the statistic is saved into the monitor if true
	 */
	template< class ValueType >
	void add(moStat<EOT, ValueType> & _stat, bool _monitoring = true) {
		checkpoint->add(_stat);

		if (_monitoring) {
			moVectorMonitor<EOT> * monitor = new moVectorMonitor<EOT>(_stat);
			monitorVec.push_back(monitor);
			checkpoint->add(*monitor);
		}
	}

	/**
	 * To sample the search and get the statistics which are stored in the moVectorMonitor vector
	 */
	void operator()(void) {
		// clear all statistic vectors
		for (unsigned i = 0; i < monitorVec.size(); i++)
			monitorVec[i]->clear();

		// change the checkpoint to compute the statistics
		localSearch->setContinuator(*checkpoint);

		// the initial solution
		EOT solution;

		// initialisation of the solution
		init(solution);

		// compute the sampling
		(*localSearch)(solution);

		// set back to initial continuator
		localSearch->setContinuator(*continuator);
	}

	/**
	 * to set the precision of the output file
	 * @param _precision precision of the output (number of digit)
	 */
	void precision(unsigned int _precision) {
	  precisionOutput = _precision;
	}

	/**
	 * to export the vectors of values into one file
	 *
	 * @param _filename file name
	 * @param _delim delimiter between statistics
	 * @param _openFile to specify if it writes at the following of the file
	 * @param _header if true, print the headers which are the name of the statistic
	 */
	  void fileExport(std::string _filename, std::string _delim = " ", bool _openFile = false, bool _header = false) {
		// create file
		std::ofstream os;

		if(! _openFile)
			os.open(_filename.c_str());
		else
			os.open(_filename.c_str(), std::ios::app);

		if (!os) {
			std::string str = "moSampling: Could not open " + _filename;
			throw std::runtime_error(str);
		}

		// set the precision of the output
		os.precision(precisionOutput);
		for (unsigned int j = 0; j < monitorVec.size(); j++) 
		  monitorVec[j]->precision(precisionOutput);

		if (!_openFile && _header) {
		  os << monitorVec[0]->longName();

		  for (unsigned int j = 1; j < monitorVec.size(); j++) {
		    os << _delim.c_str() << monitorVec[j]->longName();
		  }

		  os << std::endl ;
		}

		// all vector have the same size
		unsigned vecSize = monitorVec[0]->size();

		for (unsigned int i = 0; i < vecSize; i++) {
			os << monitorVec[0]->getValue(i);

			for (unsigned int j = 1; j < monitorVec.size(); j++) {
				os << _delim.c_str() << monitorVec[j]->getValue(i);
			}

			os << std::endl ;
		}

	}

	/**
	 * to export one vector of values into a file
	 * @param _col number of vector to print into file
	 * @param _filename file name
	 * @param _openFile to specify if it writes at the following of the file
	 */
	void fileExport(unsigned int _col, std::string _filename, bool _openFile=false) {
		if (_col >= monitorVec.size()) {
			std::string str = "moSampling: Could not export into file the vector. The index does not exists (too large)";
			throw std::runtime_error(str);
		}

		monitorVec[_col]->precision(precisionOutput);
		monitorVec[_col]->fileExport(_filename, _openFile);
	}


	/**
	 * to get one vector of values
	 * @param _numStat number of statistics to get (in the order of creation)
	 * @return the vector of value (all values are converted in double)
	 */
	const std::vector<double> & getValues(unsigned int _numStat) {
		return monitorVec[_numStat]->getValues();
	}

	/**
	 * to get one vector of solutions values
	 * @param _numStat number of statistics to get (in the order of creation)
	 * @return the vector of value (all values are converted in double)
	 */
	const std::vector<EOT> & getSolutions(unsigned int _numStat) {
		return monitorVec[_numStat]->getSolutions();
	}

	/**
	 * @return name of the class
	 */
	virtual std::string className(void) const {
		return "moSampling";
	}

protected:
	eoInit<EOT> & init;
	moLocalSearch<Neighbor> * localSearch;

	moContinuator<Neighbor> * continuator;
	moCheckpoint<Neighbor> * checkpoint;

	std::vector< moVectorMonitor<EOT> *> monitorVec;

  // precision of the output
  unsigned int precisionOutput;

};


#endif
