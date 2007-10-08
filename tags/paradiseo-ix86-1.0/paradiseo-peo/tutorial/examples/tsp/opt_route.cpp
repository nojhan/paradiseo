/* 
* <opt_route.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include "opt_route.h"

#define MAX_TRASH_LENGTH 1000
#define MAX_FIELD_LENGTH 1000
#define MAX_LINE_LENGTH 1000

static void getNextField (FILE * __f, char * __buff) {
  
  char trash [MAX_TRASH_LENGTH];  

  fscanf (__f, "%[ \t:\n]", trash); /* Discarding sep. */ 
  fscanf (__f, "%[^:\n]", __buff); /* Reading the field */
  fgetc (__f);
}

static void getLine (FILE * __f, char * __buff) {

  char trash [MAX_TRASH_LENGTH];  

  fscanf (__f, "%[ \t:\n]", trash); /* Discarding sep. */ 
  fscanf (__f, "%[^\n]", __buff); /* Reading the line */
}

static void loadBestRoute (FILE * __f) {

  opt_route.clear ();
  
  for (unsigned i = 0; i < numNodes; i ++) {
    Node node;
    fscanf (__f, "%u", & node);
    opt_route.push_back (node - 1);
  }
  int d; /* -1 ! */
  fscanf (__f, "%d", & d);
}

void loadOptimumRoute (const char * __filename) {

  FILE * f = fopen (__filename, "r");

  if (f) {
     
     printf ("Loading '%s'.\n", __filename);
     
     char field [MAX_FIELD_LENGTH];
     
     getNextField (f, field); /* Name */
     assert (strstr (field, "NAME"));
     getNextField (f, field); 
     //printf ("NAME: %s.\n", field);

          getNextField (f, field); /* Comment */
     assert (strstr (field, "COMMENT"));
     getLine (f, field);
     //     printf ("COMMENT: %s.\n", field);
     
     getNextField (f, field); /* Type */
     assert (strstr (field, "TYPE"));
     getNextField (f, field); 
     //printf ("TYPE: %s.\n", field);

     getNextField (f, field); /* Dimension */
     assert (strstr (field, "DIMENSION"));
     getNextField (f, field); 
     //     printf ("DIMENSION: %s.\n", field);
     numNodes = atoi (field);

     getNextField (f, field); /* Tour section */
     assert (strstr (field, "TOUR_SECTION"));
     loadBestRoute (f);
     
     getNextField (f, field); /* End of file */
     assert (strstr (field, "EOF"));
     //printf ("EOF.\n");
     
     printf ("The length of the best route is %u.\n", length (opt_route));
  }
   else {
     
     fprintf (stderr, "Can't open '%s'.\n", __filename); 
     exit (1);
   }
}

void loadOptimumRoute (eoParser & __parser) {
  
  /* Getting the path of the instance */
  
  eoValueParam <std :: string> param ("", "optimumTour", "Optimum tour") ;
  __parser.processParam (param) ;
  if (strlen (param.value ().c_str ()))
    loadOptimumRoute (param.value ().c_str ());
  else
    opt_route.fitness (0);
}

Route opt_route; /* Optimum route */


