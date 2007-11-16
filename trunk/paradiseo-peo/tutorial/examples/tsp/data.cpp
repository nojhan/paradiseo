/*
* <data.cpp>
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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include <utils/eoParser.h>

#include "data.h"
#include "node.h"

#define MAX_TRASH_LENGTH 1000
#define MAX_FIELD_LENGTH 1000
#define MAX_LINE_LENGTH 1000

static void getNextField (FILE * __f, char * __buff)
{

  char trash [MAX_TRASH_LENGTH];

  fscanf (__f, "%[ \t:\n]", trash); /* Discarding sep. */
  fscanf (__f, "%[^:\n]", __buff); /* Reading the field */
  fgetc (__f);
}

static void getLine (FILE * __f, char * __buff)
{

  char trash [MAX_TRASH_LENGTH];

  fscanf (__f, "%[ \t:\n]", trash); /* Discarding sep. */
  fscanf (__f, "%[^\n]", __buff); /* Reading the line */
}

void loadData (const char * __filename)
{

  FILE * f = fopen (__filename, "r");

  if (f)
    {

      printf ("Loading '%s'.\n", __filename);

      char field [MAX_FIELD_LENGTH];

      getNextField (f, field); /* Name */
      assert (strstr (field, "NAME"));
      getNextField (f, field);
      printf ("NAME: %s.\n", field);

      getNextField (f, field); /* Comment */
      assert (strstr (field, "COMMENT"));
      getLine (f, field);
      printf ("COMMENT: %s.\n", field);

      getNextField (f, field); /* Type */
      assert (strstr (field, "TYPE"));
      getNextField (f, field);
      printf ("TYPE: %s.\n", field);

      getNextField (f, field); /* Dimension */
      assert (strstr (field, "DIMENSION"));
      getNextField (f, field);
      printf ("DIMENSION: %s.\n", field);
      numNodes = atoi (field);

      getNextField (f, field); /* Edge weight type */
      assert (strstr (field, "EDGE_WEIGHT_TYPE"));
      getNextField (f, field);
      printf ("EDGE_WEIGHT_TYPE: %s.\n", field);

      getNextField (f, field); /* Node coord section */
      assert (strstr (field, "NODE_COORD_SECTION"));
      loadNodes (f);

      getNextField (f, field); /* End of file */
      assert (strstr (field, "EOF"));
      printf ("EOF.\n");
    }
  else
    {

      fprintf (stderr, "Can't open '%s'.\n", __filename);
      exit (1);
    }
}

void loadData (eoParser & __parser)
{

  /* Getting the path of the instance */

  eoValueParam <std :: string> param ("", "inst", "Path of the instance") ;
  __parser.processParam (param) ;
  loadData (param.value ().c_str ());
}
