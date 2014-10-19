/*
<qapEval.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _qapEval_h
#define _qapEval_h

#include "../../eo/eoEvalFunc.h"

/**
 * Full evaluation Function for QAP problem
 */
template< class EOT >
class QAPeval : public eoEvalFunc<EOT>
{
public:

  /*
   * Constructor from instance file
   *
   * @param _fileData the file name which contains the instance of QAP from QAPlib
   */
  QAPeval(string & _fileData) {
    fstream file(_fileData.c_str(), ios::in);

    if (!file) {
      string str = "QAPeval: Could not open file [" + _fileData + "]." ;
      throw runtime_error(str);
    }
    
    unsigned i, j;

    file >> n;
    A = new int *[n];
    B = new int *[n];
    
    for(i = 0; i < n; i++) {
      A[i] = new int[n];
      for(j = 0; j < n; j++) {
	file >> A[i][j];
      }
    }
    
    for(i = 0; i < n; i++) {
      B[i] = new int[n];
      for(j = 0; j < n; j++) 
	file >> B[i][j];
    }
    
    file.close();
  }
    
  /*
   *  default destructor
   */
  ~QAPeval() {
    unsigned i;
    
    if (A != NULL) {
      for(i = 0; i < n; i++) 
	delete[] A[i];
      delete[] A;
    }
    
    if (B != NULL) {
      for(i = 0; i < n; i++) 
	delete[] B[i];
      delete[] B;
    }
  }
  
  /*
   * full evaluation for QAP
   * 
   * @param _solution the solution to evaluate
   */
  void operator()(EOT & _solution) { 
    int cost = 0;
    
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
	cost += A[i][j] * B[_solution[i]][_solution[j]]; 
    
    _solution.fitness(cost);
  }

  /*
   * to get the matrix A
   *
   * @return matrix A
   */
  int** getA() {
    return A;
  }

  /*
   * to get the matrix B
   *
   * @return matrix B
   */
  int** getB() {
    return B;
  }

  /*
   * to get the number of objects, of variables
   *
   * @return number of objects
   */
  int getNbVar() {
    return n;
  }

private:
    // number of variables
    int n;

    // matrix A
    int ** A;

    // matrix B
    int ** B;

};

#endif
