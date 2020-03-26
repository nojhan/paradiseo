/*
<ubqpEval.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sebastien Verel

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

#ifndef _ubqpEval_h
#define _ubqpEval_h

#include <vector>
#include <eoEvalFunc.h>

/**
 * Full evaluation Function 
 * for unconstrainted binary quadratic programming problem
 */
template< class EOT >
class UbqpEval : public eoEvalFunc<EOT>
{
public:

  /**
   * Constructor
   * instance is given in the ORLIB format (0) or matrix format (1): 
   * The format of these data files is:
   * number of test problem in the serie
   * for each test problem in turn:
   *    - Format 0:
   *         number of variables (n), number of non-zero elements in the q(i,j) matrix
   *           for each non-zero element in turn: 
   *           i, j, q(i,j) {=q(j,i) as the matrix is symmetric}
   *    - Format 1:
   *         number of variables (n)
   *           for each line i
   *               for each columm j
   *                   q(i,j)
   * @param _fileName file name of the instance in ORLIB format
   * @param format id of the file format (0 or 1)
   * @param _numInstance the number of the given instance to solve
   */
  UbqpEval(std::string & _fileName, unsigned format = 0, unsigned int _numInstance = 0) {
    std::fstream file(_fileName.c_str(), std::ios::in);

    if (!file) {
      // std::string str = "UbqpEval: Could not open file [" + _fileName + "]." ;
      throw eoFileError(_fileName);
    }

    unsigned int nbInstances;
    file >> nbInstances;

    // number of non zero in the matrix
    unsigned int nbNonZero = 0;

    unsigned int i, j;
    int v;

    for(unsigned k = 0; k < _numInstance; k++) {
      if (format == 0) {
	file >> nbVar >> nbNonZero ;

	for(unsigned kk = 0; kk < nbNonZero; kk++)
	  file >> i >> j >> v;
      } else {
	file >> nbVar ;

	for(unsigned int i = 0; i < nbVar; i++) {
	  for(unsigned int j = 0; j < nbVar; j++) {
	    file >> v;
	  }
	}
      }

    }

    // the chosen instance
    if (format == 0)
      file >> nbVar >> nbNonZero ;
    else
      file >> nbVar ;

    // creation of the matrix
    Q = new int*[nbVar];

    for(unsigned int i = 0; i < nbVar; i++) {
      Q[i] = new int[nbVar];
      for(unsigned int j = 0; j < nbVar; j++)
	Q[i][j] = 0;
    }

    // read the matrix
    if (format == 0) {
      for(unsigned int k = 0; k < nbNonZero; k++) {
	file >> i >> j >> v;
	if (i > 0 && j > 0)
	  Q[i - 1][j - 1] = v;
	else {
	  std::string str = "UbqpEval: some indices are 0 in the instance file (in format 0), please check." ;
	  throw eoException(str);
	}
      }
    } else {
      for(unsigned int i = 0; i < nbVar; i++) {
	for(unsigned int j = 0; j < nbVar; j++) {
	  file >> v;
	  Q[i][j] = v;
	}
      }
    }
    
    file.close();

    // put the matrix in lower triangular form
    for(unsigned i = 1; i < nbVar; i++)
      for(unsigned int j = 0; j < i; j++) {
	Q[i][j] = Q[i][j] + Q[j][i];
	Q[j][i] = 0;
      }

  }

  /**
   * Destructor
   */
  ~UbqpEval() {
    if (Q != NULL) {
      for(unsigned i = 0; i < nbVar; i++)
	delete[] Q[i];

      // delete the matrix
      delete[] Q;
    }
  }

  /**
   * fitness evaluation of the solution
   *
   * @param _solution the solution to evaluation
   */
  virtual void operator()(EOT & _solution) {
    int fit = 0;
    unsigned int j;

    for(unsigned i = 0; i < nbVar; i++)
      if (_solution[i] == 1) 
	for(j = 0; j <= i; j++)
	  if (_solution[j] == 1) 
	    fit += Q[i][j];
	  
    _solution.fitness(fit);
  }

  /*
   * to get the matrix Q
   *
   * @return matrix Q
   */
  int** getQ() {
    return Q;
  }

  /*
   * to get the number of variable (bit string length)
   *
   * @return bit string length
   */
  int getNbVar() {
    return nbVar;
  }

  void print() {
    std::cout << nbVar << std::endl;
    for(unsigned int i = 0; i < nbVar; i++) {
      for(unsigned int j = 0; j < nbVar; j++) {
	std::cout << Q[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

private:
  /**
   * variables (used in incremental evaluation)
   */

  // number of variable
  unsigned int nbVar; 

  // matrix of flux:
  //   the matrix is put in lower triangular form: for i<j Q[i][j] = 0
  int ** Q;
};

#endif
