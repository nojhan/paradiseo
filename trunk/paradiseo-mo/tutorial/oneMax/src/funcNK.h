#ifndef __funcNK
#define __funcNK

#include <eo>
#include <fstream>
#include <vector>

using namespace std;

template< class EOT >
class FuncNK : public eoEvalFunc<EOT> {
public:
  // tables des contributions
  double ** tables;

  // liste des liens epistatiques en fonction du bit i
  // pour chaque contribution, donne la liste des variables consernés
  unsigned ** links;

  // liste inverse
  // pour chaque variable, donne la liste indices des contributions
  vector<unsigned> * knils;

  unsigned N;
  unsigned K;

  // constructeur vide
  FuncNK() : N(0), K(0)
  {
    tables = NULL;
    links  = NULL;
  };

  FuncNK(unsigned _n) : N(_n), K(0)
  {
    tables = NULL;
    links  = NULL;
  };

  // construction de tables aléatoirement
  FuncNK(int n, int k, bool consecutive = false) : N(n), K(k)
  {
    if (consecutive)
      consecutiveTables();
    else
      randomTables();
  };

  // construction à partir d'un fichier des tables et des liens
  FuncNK(const char * fichier = "")
  {
    load(fichier);
  };

  ~FuncNK()
  {
    deleteTables();
  };

  void buildTables()
  {
    links  = new unsigned*[N];
    knils  = new vector<unsigned>[N];
    tables = new double*[N];
    for(unsigned i = 0; i < N; i++) {
      tables[i] = new double[1<<(K+1)];
      links[i]  = new unsigned[K+1];
      knils[i].clear();
    }
  };

  void deleteTables()
  {
    if (links != NULL) {
      for(int i = 0; i < N; i++) {
	delete [] (links[i]);
      }
      delete [] links;
      links = NULL;
    }

    if (knils != NULL) {
      /*
      for(int i = 0; i < N; i++) {
	knils[i].clear();
      }
      */
      delete [] knils;
      knils = NULL;
    }

    if (tables != NULL) {
      for(int i = 0; i < N; i++) {
	delete [] (tables[i]);
      }
      delete [] tables;
      tables = NULL;
    }
  };

  virtual void load(const char * nomTables)
  {
    fstream file;
    file.open(nomTables, ios::in);

    if (file.is_open()) {
      //cout <<"loadTables: chargement des tables " <<nomTables <<endl;
      string s;

      // lecture des commentaires
      string line;
      file >> s;
      while (s[0] == 'c') {
	getline(file,line,'\n');
	file >> s;
      }

      // lecture des parametres
      if (s[0] != 'p') {
	cerr <<"loadTables: erreur de lecture de paramètre pour " << nomTables <<endl;
	exit(1);
      }

      file >> s;
      if (s != "NK") {
	cerr <<"erreur "  <<nomTables << " n'est pas un fichier NK" <<endl;
	exit(1);
      }

      // lecture des paramètres N et K
      file >> N >> K;
      buildTables();

      // lecture des liens
      if (s[0] != 'p') {
	cerr <<"loadTables: erreur de lecture de paramètre 'links' pour " << nomTables <<endl;
	exit(1);
      }

      file >> s;
      if (s == "links") {
	loadLinks(file);
      } else {
	cerr <<"loadTables: erreur de lecture de paramètre 'links' pour " << nomTables <<endl;
	exit(1);
      }

      // lecture des tables
      if (s[0] != 'p') {
	cerr <<"loadTables: erreur de lecture de paramètre 'tables' pour " << nomTables <<endl;
	exit(1);
      }

      file >> s;

      if (s == "tables") {
	loadTables(file);
      } else {
	cerr << "loadTables: erreur de lecture de paramètre 'tables' pour " << nomTables <<endl;
	exit(1);
      }

      file.close();
    } else {
      cerr << "loadTables: impossible d'ouvrir " << nomTables << endl;
    }

  };

  void loadLinks(fstream & file) {
    for(int j = 0; j < K+1; j++)
      for(int i = 0; i < N; i++) {
	file >> links[i][j];
	knils[links[i][j]].push_back(i);
      }
  }

  void loadTables(fstream & file) {
    for(int j = 0; j < (1<<(K+1)); j++)
      for(int i = 0; i < N; i++)
	file >> tables[i][j];
  }

  virtual void save(const char * nomTables)
  {
    //    cout <<"saveTables: sauvegarde de la table " <<nomTables <<endl;
    fstream file;
    file.open(nomTables, ios::out);

    if (file.is_open()) {
      file << "c name of file : " << nomTables << endl;
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
      cerr <<"saveTables: impossible d'ouvrir " <<nomTables <<endl;
    }
  };

  void affiche()
  {
    int j;
    for(int i=0; i<N; i++) {
      cout <<"link " <<i <<" : ";
      for(j = 0; j <= K; j++)
	cout <<links[i][j] <<" ";
      cout <<endl;
    }
    cout <<endl;

    for(int i=0; i<N; i++) {
      cout <<"knils " <<i <<" : ";
      for(j=0; j<knils[i].size(); j++)
	cout <<knils[i][j] <<" ";
      cout <<endl;
    }
    cout <<endl;

    for(int i=0; i<N; i++) {
      cout <<"table " <<i <<endl;
      for(j=0; j<(1<<(K+1)); j++)
	cout << tables[i][j] <<endl;
    }
  };

  // calcul de la fitness
  virtual void operator() (EOT & genome)
  {
    double accu = 0.0;

    for(int i = 0; i < N; i++)
      accu += tables[i][sigma(genome,i)];
    //  double M = 0.05;
    //  genome.fitness( M * ((unsigned) (accu / (N*M))) );  // affecte la fitness du genome
    genome.fitness( accu / (double) N );  // affecte la fitness du genome
  };

protected:

  void initTirage(int tabTirage[]) {
    for(int i = 0; i<N; i++)
      tabTirage[i] = i;
  };

  void perm(int tabTirage[],int i, int j) {
    int k = tabTirage[i];
    tabTirage[i] = tabTirage[j];
    tabTirage[j] = k;
  };

  void tire(int i,int tabTirage[]) {
    int t[K+1];
    for(int j=0; j<K+1; j++) {
      if (j==0) t[j]=i;
      else t[j] = rng.random(N-j);
      links[i][j] = tabTirage[t[j]];
      knils[tabTirage[t[j]]].push_back(i);
      perm(tabTirage, t[j], N-1-j);
    }
    for(int j=K; j>=0; j--)
      perm(tabTirage, t[j], N-1-j);
  };

  void consecutiveLinks(int i) {
    for(int j=0; j<K+1; j++) {
      links[i][j] = (i + j) % N;
      knils[(i + j) % N].push_back(i);
    }
  };

  // tables et liens aléatoires
  virtual void randomTables() {
    buildTables();

    int tabTirage[N];
    initTirage(tabTirage);

    for(int i = 0; i < N; i++) {
      // construit les loci epistatiquement liés au locus i
      tire(i, tabTirage);

      // la table Fi
      for(int j = 0; j < (1<<(K+1)); j++)
	tables[i][j] = rng.uniform();
    }
  };

  // tables et liens aléatoires
  virtual void consecutiveTables() {
    buildTables();

    for(int i = 0; i < N; i++) {
      // construit les loci epistatiquement liés au locus i
      consecutiveLinks(i);

      // la table Fi
      for(int j = 0; j < (1<<(K+1)); j++)
	tables[i][j] = rng.uniform();
    }
  };

  // extrait les bits liés au bit n°i
  unsigned int sigma(EOT & genome, int i)
  {
    unsigned int n    = 1;
    unsigned int accu = 0;
    for(int j=0; j<K+1; j++) {
      if (genome[links[i][j]] == 1)
	accu = accu | n;
      n = n<<1;
    }
    return accu;
  };

  };

#endif
