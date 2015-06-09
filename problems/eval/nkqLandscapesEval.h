/*
<nkqLandscapesEval.h>
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

#ifndef __nkqLandscapesEval_H
#define __nkqLandscapesEval_H

#include <eval/nkLandscapesEval.h>

/*
 * Neutral version of NK landscapes: the 'quantised' NK
 * see work of E. J. Newman, and R. Engelhardt "Effects of Neutral Selection on the Evolution of Molecular Species" (1998)
 */
template< class EOT >
class nkqLandscapesEval : public nkLandscapesEval<EOT> {
public:
  // parameter N : size of the bit string
  using nkLandscapesEval<EOT>::N;
  // parameter K : number of epistatic links
  using nkLandscapesEval<EOT>::K;
  // Table of contributions 
  using nkLandscapesEval<EOT>::tables;
  // Links between each bit
  using nkLandscapesEval<EOT>::links;

  using nkLandscapesEval<EOT>::buildTables;
  using nkLandscapesEval<EOT>::loadLinks;
  using nkLandscapesEval<EOT>::loadTables;
  using nkLandscapesEval<EOT>::consecutiveTables;
  using nkLandscapesEval<EOT>::randomTables;
  using nkLandscapesEval<EOT>::generateTables;

  // parameter q : number of different integer values in the table: [0..q[
  unsigned q;

  /**
   * Empty constructor
   */
  nkqLandscapesEval() : nkLandscapesEval<EOT>(), q(0) { }

  /**
   * Constructor of random instance
   *
   * @param _N size of the bit string
   * @param _K number of the epistatic links
   * @param consecutive : if true then the links are consecutive (i, i+1, i+2, ..., i+K), else the links are randomly choose from (1..N) 
   */
  nkqLandscapesEval(unsigned _N, unsigned _K, unsigned _q, bool consecutive = false) : nkLandscapesEval<EOT>() {
    N = _N;
    K = _K;
    q = _q;

    if (consecutive)
      consecutiveTables();
    else
      randomTables();

    generateTables();
  }

  /**
   * Constructor from a file instance
   *
   * @param _fileName the name of the file of the instance
   */
  nkqLandscapesEval(const char * _fileName) : nkLandscapesEval<EOT>(_fileName) { }

  /**
   * Load the instance from a file
   *
   * @param _fileName file name of the instance
   */
  virtual void load(const string _fileName)
  {
    fstream file;
    file.open(_fileName.c_str(), ios::in);

    if (file.is_open()) {
      string s;

      // Read the commentairies
      string line;
      file >> s;
      while (s[0] == 'c') {
	getline(file,line,'\n');
	file >> s;
      }

      // Read the parameters
      if (s[0] != 'p') {
	string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] at the begining." ;
	throw runtime_error(str);
      }

      file >> s;
      if (s != "NKq") {
	string str = "nkqLandscapesEval.load: -- NKq -- expected in [" + _fileName + "] at the begining." ;
	throw runtime_error(str);
      }

      // read parameters N, K and q
      file >> N >> K >> q;
      buildTables();

      // read the links
      if (s[0] != 'p') {
	string str = "nkqLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the parameters N, K, and q." ;
	throw runtime_error(str);
      }

      file >> s;
      if (s == "links") {
	loadLinks(file);
      } else {
	string str = "nkqLandscapesEval.load: -- links -- expected in [" + _fileName + "] after the parameters N, K, and q." ;
	throw runtime_error(str);
      }

      // lecture des tables
      if (s[0] != 'p') {
	string str = "nkqLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the links." ;
	throw runtime_error(str);
     }

      file >> s;

      if (s == "tables") {
	loadTables(file);
      } else {
	string str = "nkqLandscapesEval.load: -- tables -- expected in [" + _fileName + "] after the links." ;
	throw runtime_error(str);
      }

      file.close();
    } else {
	string str = "nkqLandscapesEval.load: Could not open file [" + _fileName + "]." ;
	throw runtime_error(str);
    }

  };

  /**
   * Save the current intance into a file
   * 
   * @param _fileName the file name of instance
   */
  virtual void save(const char * _fileName) {
    fstream file;
    file.open(_fileName, ios::out);

    if (file.is_open()) {
      file << "c name of the file : " << _fileName << endl;
      file << "p NKq " << N << " " << K << " " << q << endl;

      file << "p links" << endl;
      for(int j=0; j<K+1; j++)
	for(int i=0; i<N; i++)
	  file << links[i][j] << endl;

      file << "p tables" << endl;
      for(int j=0; j<(1<<(K+1)); j++) {
	for(int i=0; i<N; i++)
	  file << tables[i][j] << " ";
	file << endl;
      }
      file.close();
    } else {
      string fname(_fileName);
      string str = "nkqLandscapesEval.save: Could not open file [" + fname + "]." ;
      throw runtime_error(str);
    }
  };

protected:

  /**
   * To generate a contribution in the table f_i 
   *
   */
  virtual double contribution() {
    return ((double) rng.random(q)) / (double) (q-1);
  }
  
 
};

#endif
