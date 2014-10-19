/*
<nkLandscapesEval.h>
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

#ifndef __nkLandscapesEval_H
#define __nkLandscapesEval_H

#include "../../eo/eoEvalFunc.h"

template< class EOT >
class nkLandscapesEval : public eoEvalFunc<EOT> {
public:
  // parameter N : size of the bit string
  unsigned N;

  // parameter K : number of epistatic links
  unsigned K;

  // Table of contributions 
  double ** tables;
    
  // Links between each bit
  // links[i][0], ..., links[i][K] : the (K+1) links to the bit i
  unsigned ** links;

  /**
   * Empty constructor
   */
  nkLandscapesEval() : N(0), K(0) 
  {
    tables = NULL;
    links  = NULL;
  };

  /**
   * Constructor of random instance
   *
   * @param _N size of the bit string
   * @param _K number of the epistatic links
   * @param consecutive : if true then the links are consecutive (i, i+1, i+2, ..., i+K), else the links are randomly choose from (1..N) 
   */
  nkLandscapesEval(int _N, int _K, bool consecutive = false) : N(_N), K(_K) 
  {
    if (consecutive)
      consecutiveTables();
    else
      randomTables();
  };

  /**
   * Constructor from a file instance
   *
   * @param _fileName the name of the file of the instance
   */
  nkLandscapesEval(const char * _fileName)
  { 
    string fname(_fileName);
    load(fname);
  };

  /**
   * Default destructor of the table contribution and the links
   */
  ~nkLandscapesEval() 
  {
    deleteTables();
  };

  /**
   * Reserve the space memory for the links and the table
   */
  void buildTables()
  {
    links  = new unsigned*[N];
    tables = new double*[N];

    for(unsigned i = 0; i < N; i++) {
      tables[i] = new double[1<<(K+1)];
      links[i]  = new unsigned[K+1];
    }
  };

  /**
   * Free the space memory of the table contributions and the links
   */
  void deleteTables()
  {
    if (links != NULL) {
      for(int i = 0; i < N; i++) {
	delete [] (links[i]);
      }
      delete [] links;
      links = NULL;
    }

    if (tables != NULL) {
      for(int i = 0; i < N; i++) {
	delete [] (tables[i]);
      }
      delete [] tables;
      tables = NULL;
    }
  };

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
      if (s != "NK") {
	string str = "nkLandscapesEval.load: -- NK -- expected in [" + _fileName + "] at the begining." ;
	throw runtime_error(str);
      }

      // read parameters N and K
      file >> N >> K;
      buildTables();

      // read the links
      if (s[0] != 'p') {
	string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the parameters N and K." ;
	throw runtime_error(str);
      }

      file >> s;
      if (s == "links") {
	loadLinks(file);
      } else {
	string str = "nkLandscapesEval.load: -- links -- expected in [" + _fileName + "] after the parameters N and K." ;
	throw runtime_error(str);
      }

      // lecture des tables
      if (s[0] != 'p') {
	string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the links." ;
	throw runtime_error(str);
     }

      file >> s;

      if (s == "tables") {
	loadTables(file);
      } else {
	string str = "nkLandscapesEval.load: -- tables -- expected in [" + _fileName + "] after the links." ;
	throw runtime_error(str);
      }

      file.close();
    } else {
	string str = "nkLandscapesEval.load: Could not open file [" + _fileName + "]." ;
	throw runtime_error(str);
    }

  };

  /**
   * Read the links from the file
   *
   * @param file the file to read 
   */
  void loadLinks(fstream & file) {
    for(int j = 0; j < K+1; j++)
      for(int i = 0; i < N; i++) {
	file >> links[i][j];
      }
  }

  /**
   * Read the tables from the file
   *
   * @param file the file to read 
   */
  void loadTables(fstream & file) {
    for(int j = 0; j < (1<<(K+1)); j++)
      for(int i = 0; i < N; i++)
	file >> tables[i][j];
  }

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
      file << "p NK " << N << " " << K <<endl;

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
      string str = "nkLandscapesEval.save: Could not open file [" + fname + "]." ;
      throw runtime_error(str);
    }
  };

  /**
   * Print the instance to the screen
   */
  void print() {
    int j;
    for(int i=0; i<N; i++) {
      std::cout <<"link " <<i <<" : ";
      for(j = 0; j <= K; j++) 
	std::cout <<links[i][j] <<" ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    for(int i=0; i<N; i++) {
      std::cout <<"table " << i << std::endl;
      for(j=0; j<(1<<(K+1)); j++)
	std::cout << tables[i][j] << std::endl;
    }
  };

  /**
   * Compute the fitness value
   *
   * @param _solution the solution to evaluate
   */
  virtual void operator() (EOT & _solution) {
    double accu = 0.0;
  
    for(int i = 0; i < N; i++)
      accu += tables[ i ][ sigma(_solution, i) ];

    _solution.fitness( accu / (double) N );  
  }

protected:
    
  /**
   * Compute the mask of the linked bits
   *
   * @param _solution the solution to evaluate
   * @param i the bit of the contribution 
   */
  unsigned int sigma(EOT & _solution, int i) {
    unsigned int n    = 1;
    unsigned int accu = 0;

    for(int j = 0; j < K+1; j++) {
      if (_solution[ links[i][j] ] == 1)
	accu = accu | n;
      n = n << 1;
    }

    return accu;
  }

  /**
   * To generate random instance without replacement : initialization
   *
   * @param tabTirage the table to initialize
   */
  void initTirage(int tabTirage[]) {
    for(int i = 0; i<N; i++)
      tabTirage[i] = i;
  };

  /**
   * To generate random instance without replacement : swap
   *
   * @param tabTirage the table of bits
   * @param i  first indice to swap
   * @param j second indice to swap
   */
  void perm(int tabTirage[],int i, int j) {
    int k = tabTirage[i];
    tabTirage[i] = tabTirage[j];
    tabTirage[j] = k;
  };

  /**
   * To generate random instance without replacement 
   * choose the linked bit without replacement
   *
   * @param i the bit of contribution
   * @param tabTirage the table of bits
   */
  void choose(int i, int tabTirage[]) {
    int t[K+1];
    for(int j=0; j<K+1; j++) {
      if (j==0) t[j]=i;
      else t[j] = rng.random(N-j);
      links[i][j] = tabTirage[t[j]];
      perm(tabTirage, t[j], N-1-j);
    }
    for(int j=K; j>=0; j--)
      perm(tabTirage, t[j], N-1-j);
  }

  /**
   * To generate an instance with no-random links
   *
   * @param i the bit of contribution
   */
  void consecutiveLinks(int i) {
    for(int j = 0; j < K+1; j++) {
      links[i][j] = (i + j) % N;
    }
  }

  /**
   * To generate a contribution in the table f_i 
   *
   */
  virtual double contribution() {
    return rng.uniform();
  }

  /**
   * To generate instance with random (without replacement) links
   *
   */
  virtual void randomTables() {
    buildTables();
      
    int tabTirage[N];
    initTirage(tabTirage);

    for(int i = 0; i < N; i++) {
      // random links to the bit
      choose(i, tabTirage);  
      
      // table of contribution with random numbers from [0,1)
      for(int j = 0; j < (1<<(K+1)); j++) 
	tables[i][j] = contribution();
    }
  }
 
  /**
   * To generate instance with consecutive links
   *
   */
  virtual void consecutiveTables() {
    buildTables();
      
    for(int i = 0; i < N; i++) {
      // consecutive link to bit i
      consecutiveLinks(i);  
      
      // table of contribution with random numbers from [0,1)
      for(int j = 0; j < (1<<(K+1)); j++) 
	tables[i][j] = contribution();
    }
  }
 
};

#endif
