/*
<maxSATeval.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

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

#ifndef _maxSATeval_h
#define _maxSATeval_h

#include <vector>
#include "../../eo/eoEvalFunc.h"

/**
 * Full evaluation Function for max-SAT problem
 */
template< class EOT >
class MaxSATeval : public eoEvalFunc<EOT>
{
public:

  /**
   * Constructor
   * generate a random instance of max k-SAT problem
   *
   * @param _n number of variables
   * @param _m number of clauses
   * @param _k of litteral by clause
   */
  MaxSATeval(unsigned _n, unsigned _m, unsigned _k) {
    nbVar      = _n;
    nbClauses  = _m;
    nbLitteral = _k;

    // creation of the clauses
    clauses = new std::vector<int>[nbClauses];

    // the variables are numbered from 1 to nbVar
    variables = new std::vector<int>[nbVar + 1];

    //int var[nbVar];
    std::vector<int> var;
    var.resize(nbVar);
    unsigned i, j, ind;

    // to selected nbLitteral different variables in the clauses
    for(i = 0; i < nbVar; i++)
      var[i] = i + 1;
    
    int number, tmp;

    // generation of the clauses
    for(i = 0; i < nbClauses; i++) {
      for(j = 0 ; j < nbLitteral; j++) {
	// selection of the variable
	ind = rng.random(nbVar - j);
	number = var[ind];

	// permutation for forbidd identical variables
	tmp = var[ind];
	var[ind] = var[nbVar - j - 1];
	var[nbVar - j - 1] = tmp;

	// litteral = (variable) or (not variable) ?
	// negative value means not variable
	if (rng.flip())
	  number = -number;
	
	// variable number belong to clause i
	if (number < 0)
	  variables[-number].push_back(-i);
	else
	  variables[number].push_back(i);

	// clause i has got the litteral number 
	clauses[i].push_back(number);
      }
    }

  }

  /**
   * Constructor
   * instance is given in the cnf format (see dimacs)
   *
   * @param _fileName file name of the instance in cnf format
   */
  MaxSATeval(std::string & _fileName) {
    std::fstream file(_fileName.c_str(), std::ios::in);

    if (!file) {
      std::string str = "MaxSATeval: Could not open file [" + _fileName + "]." ;
      throw std::runtime_error(str);
    }
    
    std::string s;

    // commentaries
    std::string line;
    file >> s;
    while (s[0] == 'c') {
      getline(file,line,'\n');
      file >> s;
    }

    // parameters
    if (s[0] != 'p') {
      std::string str = "MaxSATeval: could not read the parameters of the instance from file [" + _fileName + "]." ;
      throw std::runtime_error(str);
    }
    file >> s;
    if (s != "cnf") {
      std::string str = "MaxSATeval: " + _fileName + " is not a file in cnf format.";
      throw std::runtime_error(str);
    }

    file >> nbVar >> nbClauses;
    nbLitteral = 0; // could be different from one clause to antoher, so no value

    // creation of the clauses
    clauses = new std::vector<int>[nbClauses];

    // the variables are numbered from 1 to nbVar
    variables = new std::vector<int>[nbVar + 1];

    // read the clauses
    int number;
    for(unsigned i = 0; i < nbClauses; i++) {
      do {
	file >> number;
	if (number != 0) {
	  clauses[i].push_back(number);
	  if (number < 0)
	    number = -number;

	  if (number < 0)
	    variables[-number].push_back(-i);
	  else
	    variables[number].push_back(i);
	}
      } while (number != 0);    
    }
    
    file.close();
  }

  /**
   * Destructor
   */
  ~MaxSATeval() {
    // delete the clauses
    delete[] clauses;

    // delete the variables
    delete[] variables;
  }

  /**
   * export the instance to a file in cnf format
   *
   * @param _fileName file name to export the instance
   */
  void save(std::string & _fileName) {
    std::fstream file(_fileName.c_str(), std::ios::out);

    if (!file) {
      std::string str = "MaxSATeval: Could not open " + _fileName;
      throw std::runtime_error(str);
    }
    
    // write some commentaries
    file << "c random max k-SAT generated by maxSATeval from paradisEO framework on sourceForge" << std::endl;
    file << "c  "<< std::endl;

    // write the parameters
    file << "p cnf " << nbVar << " " << nbClauses << std::endl;

    // write the clauses
    unsigned int i, j;

    for(i = 0; i < nbClauses; i++) {
      j = 0;
      while (j < clauses[i].size()) {
	file << clauses[i][j] << " ";
	j++;
      }    
      file << "0" << std::endl;
    }

    file.close();
  }

  /**
   * evaluation the clause
   *
   * @param _n number of the given clause
   * @param _solution the solution to evaluation
   * @return true when the clause is true
   */
  bool clauseEval(unsigned int _n, EOT & _solution) {
    unsigned nLitteral = clauses[_n].size();
    int litteral;

    bool clause = false;

    unsigned int j = 0;
    while (j < nLitteral && !clause) {
      litteral = clauses[_n][j];
      clause = ((litteral > 0) && _solution[litteral - 1]) || ((litteral < 0) && !(_solution[-litteral - 1]));
      
      j++;
    }

    return clause;
  }

  /**
   * fitness evaluation of the solution
   *
   * @param _solution the solution to evaluation
   */
  virtual void operator()(EOT & _solution) {
    unsigned int fit = 0;

    for(unsigned i = 0; i < nbClauses; i++)
      if (clauseEval(i, _solution))
	fit++;

    _solution.fitness(fit);
  }

  /**
   * Public variables (used in incremental evaluation)
   */

  // number of variables
  unsigned int nbVar; 
  // number of clauses
  unsigned int nbClauses;
  // number of litteral by clause (0 when the instance is read from file)
  unsigned int nbLitteral;

  // list of clauses: 
  //   each clause has the number of the variable (from 1 to nbVar) 
  //   when the value is negative, litteral = not(variable)
  std::vector<int> * clauses;

  // list of variables:
  //   for each variable, the list of clauses
  //   when the value is negative, litteral = not(variable) in this clause
  std::vector<int> * variables;

};

#endif
