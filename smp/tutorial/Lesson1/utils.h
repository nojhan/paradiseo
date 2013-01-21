/*
<utils.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

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

#ifndef _UTILS_H
#define _UTILS_H
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

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
