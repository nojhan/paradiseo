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

#include <eoEvalFunc.h>
#include <fstream>

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

    generateTables();
  };

  /**
   * Constructor from a file instance
   *
   * @param _fileName the name of the file of the instance
   */
  nkLandscapesEval(const char * _fileName)
  { 
    std::string fname(_fileName);
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
      for(unsigned i = 0; i < N; i++) {
	delete [] (links[i]);
      }
      delete [] links;
      links = NULL;
    }

    if (tables != NULL) {
      for(unsigned i = 0; i < N; i++) {
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
  virtual void load(const std::string _fileName)
  {
    std::fstream file;
    file.open(_fileName.c_str(), std::fstream::in);

    if (file.is_open()) {
      std::string s;

      // Read the commentairies
      std::string line;
      file >> s;
      while (s[0] == 'c') {
	getline(file,line,'\n');
	file >> s;
      }

      // Read the parameters
      if (s[0] != 'p') {
	std::string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] at the begining." ;
	throw eoException(str);
      }

      file >> s;
      if (s != "NK") {
	std::string str = "nkLandscapesEval.load: -- NK -- expected in [" + _fileName + "] at the begining." ;
	throw eoException(str);
      }

      // read parameters N and K
      file >> N >> K;
      buildTables();

      // read the links
      if (s[0] != 'p') {
	std::string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the parameters N and K." ;
	throw eoException(str);
      }

      file >> s;
      if (s == "links") {
	loadLinks(file);
      } else {
	std::string str = "nkLandscapesEval.load: -- links -- expected in [" + _fileName + "] after the parameters N and K." ;
	throw eoException(str);
      }

      // lecture des tables
      if (s[0] != 'p') {
	std::string str = "nkLandscapesEval.load: -- p -- expected in [" + _fileName + "] after the links." ;
	throw eoException(str);
     }

      file >> s;

      if (s == "tables") {
	loadTables(file);
      } else {
	std::string str = "nkLandscapesEval.load: -- tables -- expected in [" + _fileName + "] after the links." ;
	throw eoException(str);
      }

      file.close();
    } else {
	// std::string str = "nkLandscapesEval.load: Could not open file [" + _fileName + "]." ;
	throw eoFileError(_fileName);
    }

  };

  /**
   * Read the links from the file
   *
   * @param file the file to read 
   */
  void loadLinks(std::fstream & file) {
    for(unsigned j = 0; j < K+1; j++)
      for(unsigned i = 0; i < N; i++) {
	file >> links[i][j];
      }
  }

  /**
   * Read the tables from the file
   *
   * @param file the file to read 
   */
  void loadTables(std::fstream & file) {
    for(unsigned j = 0; j < (1<<(K+1)); j++)
      for(unsigned i = 0; i < N; i++)
	file >> tables[i][j];
  }

  /**
   * Save the current intance into a file
   * 
   * @param _fileName the file name of instance
   */
  virtual void save(const char * _fileName) {
    std::fstream file;
    file.open(_fileName, std::fstream::out);

    if (file.is_open()) {
      file << "c name of the file : " << _fileName << std::endl;
      file << "p NK " << N << " " << K <<std::endl;

      file << "p links" << std::endl;
      for(unsigned j=0; j<K+1; j++)
	for(unsigned i=0; i<N; i++)
	  file << links[i][j] << std::endl;

      file << "p tables" << std::endl;
      for(unsigned j=0; j<(1<<(K+1)); j++) {
	for(unsigned i=0; i<N; i++)
	  file << tables[i][j] << " ";
	file << std::endl;
      }
      file.close();
    } else {
      std::string fname(_fileName);
      std::string str = "nkLandscapesEval.save: Could not open file [" + fname + "]." ;
      throw std::runtime_error(str);
    }
  };

  /**
   * Print the instance to the screen
   */
  void print() {
    int j;
    for(unsigned i=0; i<N; i++) {
      std::cout <<"link " <<i <<" : ";
      for(j = 0; j <= K; j++) 
	std::cout <<links[i][j] <<" ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    for(unsigned i=0; i<N; i++) {
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
  
    for(unsigned i = 0; i < N; i++)
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

    for(unsigned j = 0; j < K+1; j++) {
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
    for(unsigned i = 0; i<N; i++)
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
    for(unsigned j=0; j<K+1; j++) {
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
    for(unsigned j = 0; j < K+1; j++) {
      links[i][j] = (i + j) % N;
    }
  }

  /**
   * To generate the tables:
   * The component function is random 
   * each contribution is independent from the others ones
   * and drawn from the distribution given by contribution()
   *
   */
  virtual void generateTables() {
    for(unsigned i = 0; i < N; i++) {
      for(unsigned j = 0; j < (1<<(K+1)); j++) 
	tables[i][j] = contribution();
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

    for(unsigned i = 0; i < N; i++) {
      // random links to the bit
      choose(i, tabTirage);  
    }
  }
 
  /**
   * To generate instance with consecutive links
   *
   */
  virtual void consecutiveTables() {
    buildTables();
      
    for(unsigned i = 0; i < N; i++) {
      // consecutive link to bit i
      consecutiveLinks(i);  
    }
  }
 
};

#endif
