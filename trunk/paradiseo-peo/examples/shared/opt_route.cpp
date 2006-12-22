// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "opt_route.cpp"

// (c) OPAC Team, LIFL, January 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
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


