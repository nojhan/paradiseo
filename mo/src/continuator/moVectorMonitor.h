/*
  <moVectorMonitor.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef moVectorMonitor_h
#define moVectorMonitor_h

#include <fstream>
#include <paradiseo/eo/utils/eoMonitor.h>
#include <paradiseo/eo/utils/eoParam.h>

/**
 * To save the values of the same type (double, unsigned int, or EOT) in a vector
 * It is similar to eoFileMonitor
 *
 */
template <class EOT>
class moVectorMonitor : public eoMonitor
{
public:

	/**
	 * Constructor
	 * @param _param the parameter of type double to save in the vector
	 */
	moVectorMonitor(eoValueParam<double> & _param) : doubleParam(&_param), intParam(NULL), intLongParam(NULL), intLongLongParam(NULL), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type unsigned int to save in the vector
	 */
	moVectorMonitor(eoValueParam<unsigned int> & _param) : doubleParam(NULL), intParam(&_param), intLongParam(NULL), intLongLongParam(NULL), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type unsigned int to save in the vector
	 */
	moVectorMonitor(eoValueParam<unsigned long> & _param) : doubleParam(NULL), intParam(NULL), intLongParam(&_param), intLongLongParam(NULL), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type unsigned int to save in the vector
	 */
	moVectorMonitor(eoValueParam<long long int> & _param) : doubleParam(NULL), intParam(NULL), intLongParam(NULL), intLongLongParam(&_param), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type EOT to save in the vector
	 */
	moVectorMonitor(eoValueParam<EOT> & _param) : doubleParam(NULL), intParam(NULL), intLongParam(NULL), intLongLongParam(NULL), eotParam(&_param)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type eoScalarFitness to save in the vector
	 */
	template <class Compare>
	moVectorMonitor(eoValueParam<eoScalarFitness<long long int, Compare> > & _param) : doubleParam(NULL), intParam(NULL), intLongParam(NULL), intLongLongParam(& (eoValueParam<long long int>&)_param), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param the parameter of type eoScalarFitness to save in the vector
	 */
	template <class ScalarType, class Compare>
	moVectorMonitor(eoValueParam<eoScalarFitness<ScalarType, Compare> > & _param) : doubleParam( & (eoValueParam<double>&)_param), intParam(NULL), intLongParam(NULL), intLongLongParam(NULL), eotParam(NULL)
	{ 
		// precision of the output by default
		precisionOutput = std::cout.precision();
	}

	/**
	 * Default Constructor
	 * @param _param unvalid Parameter
	 */
	template <class T>
	moVectorMonitor(eoValueParam<T> & _param) : doubleParam(NULL), intParam(NULL), intLongParam(NULL), intLongLongParam(NULL), eotParam(NULL)
	{
		std::cerr << "Sorry the type can not be in a vector of moVectorMonitor" << std::endl;
	}

	/**
	 * To test if the value are basic type (double or unsigned int), or EOT type
	 *
	 * @return true if the type is a EOT type
	 */
	bool solutionType() {
		return eotParam != NULL;
	}

	/**
	 * To "print" the value of the parameter in the vector
	 *
	 * @return this monitor (sorry I don't why, but it is like this in EO)
	 */
	eoMonitor& operator()(void) {
	  if (doubleParam != NULL) 
	    valueVec.push_back(doubleParam->value()); 
	  else
	    if (intParam != NULL)
	      valueVec.push_back((double) intParam->value());
	    else
	      if (intLongParam != NULL)
		valueVec.push_back((double) intLongParam->value());
	      else
		if (intLongLongParam != NULL) 
		  valueVec.push_back((double) intLongLongParam->value());
		else 
		  eotVec.push_back(eotParam->value());
	  return *this ;
	}

	/**
	 * To have all the values
	 *
	 * @return the vector of values
	 */
	const std::vector<double>& getValues() const {
		return valueVec;
	}

	/**
	 * To have all the solutions
	 *
	 * @return the vector of solutions
	 */
	const std::vector<EOT>& getSolutions() const {
		return eotVec;
	}

	/**
	 * to get the value out.
	 * @return the string of the value
	 */
	std::string getValue(unsigned int i) const {
		std::ostringstream os;

		// set the precision of the output
		os.precision(precisionOutput);

		if (eotParam == NULL)
			os << (valueVec[i]) ;
		else
			os << (eotVec[i]) ;

		return os.str();
	}

	/**
	 * Returns the long name of the statistic (which is a eoParam) 
	 *
	 * @return longName of the statistic
	 */
	const std::string& longName() const { 
	  if (doubleParam != NULL) 
	    return doubleParam->longName(); 
	  else
	    if (intParam != NULL)
	      return intParam->longName();
	    else
	      if (intLongParam != NULL)
		return intLongParam->longName();
	      else 
		if (intLongLongParam != NULL)
		  return intLongLongParam->longName();
		else 
		  return eotParam->longName();
	}

	/**
	 * clear the vector
	 */
	void clear() {
		valueVec.clear();
		eotVec.clear();
	}

	/**
	 * number of value
	 * @return size of the vector
	 */
	unsigned int size() {
		if (eotParam == NULL)
			return valueVec.size();
		else
			return eotVec.size();
	}

	/**
	 * to set the precision of the output file
	 * @param _precision precision of the output (number of digit)
	 */
	void precision(unsigned int _precision) {
	  precisionOutput = _precision;
	}

	/**
	 * to export the vector of values into one file
	 * @param _filename file name
	 * @param _openFile to specify if it writes at the following of the file
	 */
	void fileExport(std::string _filename, bool _openFile=false) {
		// create file
		std::ofstream os;

		if(! _openFile)
			os.open(_filename.c_str());

		else
			os.open(_filename.c_str(),std::ios::app);


		if (!os) {
			std::string str = "moVectorMonitor: Could not open " + _filename;
			throw std::runtime_error(str);
		}

		for (unsigned int i = 0; i < size(); i++) {
			os << getValue(i);

			os << std::endl ;
		}

	}

	/**
	 * @return name of the class
	 */
	virtual std::string className(void) const {
		return "moVectorMonitor";
	}

protected:
	eoValueParam<double> * doubleParam ;
	eoValueParam<unsigned int> * intParam ;
	eoValueParam<unsigned long> * intLongParam ;
	eoValueParam<long long int> * intLongLongParam ;
	eoValueParam<EOT> * eotParam ;

	std::vector<double> valueVec;
	std::vector<EOT> eotVec;

  // precision of the output
  unsigned int precisionOutput;

};


#endif
