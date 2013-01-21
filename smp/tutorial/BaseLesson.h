#ifndef _BASELESSON_H
#define _BASELESSON_H
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

int n = 10;
int** a;
int** b;

int bkv; //best known value

struct parameters
{
  unsigned seed ;
  int popSize;
  int tSize;
  string inst;
  string loadName;
  string schema;
  double pCross;
  double pMut;
  int minGen;
  int maxGen;
};

class Indi : public EO<eoMinimizingFitness> { 

public:   
 
  int* solution; 
  int evalNb = 0;

  Indi () {
    solution = new int[n];
    create();
  } 

  Indi (const Indi & _problem){ // copy constructor
    solution = new int[n];
    for (int i = 0; i < n ; i++){
      solution[i] = _problem.solution[i];
    }
    if (!_problem.invalid()) // if the solution has already been evaluated
      fitness(_problem.fitness()); // copy the fitness 
  }

  ~Indi(){ // destructor
    delete[] solution;
  }

  void operator= (const Indi & _problem){ // copy assignment operator
    for (int i = 0; i < n ; i++){
      solution[i] = _problem.solution[i];
    }
    fitness(_problem.fitness()); // copy the fitness 
  }  
  
  int& operator[] (unsigned i)
  {
  	return solution[i];	
  }

 
  void create(){ // create and initialize a solution
    int random, temp;
    for (int i=0; i< n; i++) 
      solution[i]=i;
    
    // we want a random permutation so we shuffle
    for (int i = 0; i < n; i++){
      random = rand()%(n-i) + i;
      temp = solution[i];
      solution[i] = solution[random];
      solution[random] = temp;
    }
  }

  int evaluate() { // evaluate the solution
    int cost=0;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
	      cost += a[i][j] * b[solution[i]][solution[j]];
        evalNb++;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
	    //std::cout << "Evaluation " << evalNb << std::endl;
	    
    return cost;
  }


  void printSolution() {
   for (int i = 0; i < n ; i++)
     std::cout << solution[i] << " " ;
 
   std::cout << std::endl;
  }
 
 
};
class IndiInit : public eoInit<Indi>
{
public:

  void operator()(Indi & _problem)
  {
    _problem.create();
  }
};

class IndiEvalFunc : public eoEvalFunc<Indi>
{
public:

  void operator()(Indi & _problem)
  {
    _problem.fitness(_problem.evaluate());
      
  }
};

class IndiXover : public eoQuadOp<Indi> {
public:

  /* The two parameters in input are the parents.
     The first parameter is also the output ie the child 
  */
  bool operator()(Indi & _problem1, Indi & _problem2)
  {
    int i;
    int random, temp;
    std::vector<int> unassigned_positions(n);
    std::vector<int> remaining_items(n);
    int j = 0;
				
    /* 1) find the items assigned in different positions for the 2 parents */
    for (i = 0 ; i < n ; i++){
      if (_problem1.solution[i] != _problem2.solution[i]){
	unassigned_positions[j] = i;
	remaining_items[j] = _problem1.solution[i];
	j++;
      }
    }
    
    /* 2) shuffle the remaining items to ensure that remaining items 
       will be assigned at random positions */
    for (i = 0; i < j; i++){
      random = rand()%(j-i) + i;
      temp = remaining_items[i];
      remaining_items[i] = remaining_items[random];
      remaining_items[random] = temp;
    }
						    					   
    /* 3) copy the shuffled remaining items at unassigned positions */
    for (i = 0; i < j ; i++)
      _problem1.solution[unassigned_positions[i]] = remaining_items[i];

    // crossover in our case is always possible
    return true; 
  }
};

class IndiSwapMutation: public eoMonOp<Indi> {
public:
 
  bool operator()(Indi& _problem)  {
    int i,j;
    int temp;

    // generate two different indices
    i=rand()%n;
    do 
      j = rand()%n; 
    while (i == j);  
		   
    // swap
    temp = _problem.solution[i];
    _problem.solution[i] = _problem.solution[j];
    _problem.solution[j] = temp;

    return true;
    
  }

};

void parseFile(eoParser & parser, parameters & param)
{

  // For each parameter, you can in on single line
  // define the parameter, read it through the parser, and assign it

  param.seed = parser.createParam(unsigned(time(0)), "seed", "Random number seed", 'S').value(); // will be in default section General

  // init and stop
  param.loadName = parser.createParam(string(""), "Load","A save file to restart from",'L', "Persistence" ).value();
  
  param.inst = parser.createParam(string(""), "inst","a dat file to read instances from",'i', "Persistence" ).value();
  
  param.schema = parser.createParam(string(""), "schema","an xml file mapping process",'s', "Persistence" ).value();

  param.popSize = parser.createParam(unsigned(10), "popSize", "Population size",'P', "Evolution engine" ).value();
  
  param.tSize = parser.createParam(unsigned(2), "tSize", "Tournament size",'T', "Evolution Engine" ).value();
  
  param.minGen = parser.createParam(unsigned(100), "minGen", "Minimum number of iterations",'g', "Stopping criterion" ).value();

  param.maxGen = parser.createParam(unsigned(300), "maxGen", "Maximum number of iterations",'G', "Stopping criterion" ).value();
  
  param.pCross = parser.createParam(double(0.6), "pCross", "Probability of Crossover", 'C', "Genetic Operators" ).value();
  
  param.pMut = parser.createParam(double(0.1), "pMut", "Probability of Mutation", 'M', "Genetic Operators" ).value();


  // the name of the "status" file where all actual parameter values will be saved
  string str_status = parser.ProgramName() + ".status"; // default value
  string statusName = parser.createParam(str_status, "status","Status file",'S', "Persistence" ).value();

  // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
  // i.e. in case you need parameters somewhere else, postpone these
  if (parser.userNeedsHelp())
    {
      parser.printHelp(cout);
      exit(1);
    }
  if (statusName != "")
    {
      ofstream os(statusName.c_str());
      os << parser;		// and you can use that file as parameter file
    }
}

void loadInstances(const char* filename, int& n, int& bkv, int** & a, int** & b) 
{
	
  ifstream data_file;       
  int i, j;
  data_file.open(filename);
  if (! data_file.is_open())
    {
      cout << "\n Error while reading the file " << filename << ". Please check if it exists !" << endl;
      exit (1);
    }
  data_file >> n;
  data_file >> bkv; // best known value
  // ****************** dynamic memory allocation ****************** /
  a = new int* [n];
  b = new int* [n];
  for (i = 0; i < n; i++) 
  {
    a[i] = new int[n];
    b[i] = new int[n];
  }

  // ************** read flows and distanceMatrixs matrices ************** /
  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++)
      data_file >> a[i][j];

  for (i = 0; i < n; i++) 
    for (j = 0; j < n; j++)
      data_file >> b[i][j];

  data_file.close();
}

#endif
