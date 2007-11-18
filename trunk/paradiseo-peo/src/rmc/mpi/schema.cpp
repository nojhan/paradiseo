/* 
* <schema.cpp>
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

#include <iostream>
#include <set>
#include <assert.h>

#include "schema.h"
#include "xml_parser.h"
#include "comm.h"
#include "node.h"
#include "../../core/peo_debug.h"


std :: vector <Node> the_schema;

Node * my_node;

static unsigned maxSpecifiedRunnerID = 0;


RANK_ID getRankOfRunner (RUNNER_ID __key) {

  for (unsigned i = 0; i < the_schema.size (); i ++)
    for (unsigned j = 0; j < the_schema [i].id_run.size (); j ++)
      if (the_schema [i].id_run [j] == __key)
        return the_schema [i].rk;
  assert (false);
  return 0; 
}

static void loadNode (int __rk_sched) {

  Node node;

  node.rk_sched = __rk_sched;

  /* ATT: name*/
  node.rk = getRankFromName (getAttributeValue ("name"));
  /* ATT: num_workers */
  node.num_workers = atoi (getAttributeValue ("num_workers").c_str ());

  while (true) {

    /* TAG: <runner> | </node> */
    std :: string name = getNextNode ();
    assert (name == "runner" || name == "node");    
    if (name == "runner") {
      /* TAG: </node> */
      node.id_run.push_back (atoi (getNextNode ().c_str ()));
      if ( node.id_run.back() > maxSpecifiedRunnerID )
        maxSpecifiedRunnerID = node.id_run.back();
      /* TAG: </runner> */
      assert (getNextNode () == "runner");
    }
    else {
      /* TAG: </node> */
      node.execution_id_run = node.id_run;
      the_schema.push_back (node); 
      break;
    }
  }
}

static void loadGroup () {

  std :: string name;

  /* ATT: scheduler*/
  int rk_sched = getRankFromName (getAttributeValue ("scheduler"));

  while (true) {

    /* TAG: <node> | </group> */
    name = getNextNode ();
    assert (name == "node" || name == "group");    
    if (name == "node")
      /* TAG: <node> */
      loadNode (rk_sched);
    else
      /* TAG: </group> */
      break;
  }
}

bool isScheduleNode () {

  return my_node -> rk == my_node -> rk_sched;
}

void loadSchema (const char * __filename) {

  openXMLDocument (__filename);

  std :: string name;

  /* TAG: <schema> */
  name = getNextNode ();
  assert (name == "schema");

  maxSpecifiedRunnerID = 0;


  while (true) {

    /* TAG: <group> | </schema> */
    name = getNextNode ();
    assert (name == "group" || name == "schema");
    if (name == "group")
      /* TAG: <group> */
      loadGroup ();
    else
      /* TAG: </schema> */
      break;
  }


  std :: set<unsigned> uniqueRunnerIDs; unsigned nbUniqueIDs = 0;
  for (unsigned i = 0; i < the_schema.size (); i ++) {
    for (unsigned j = 0; j < the_schema [i].id_run.size(); j ++) {
      uniqueRunnerIDs.insert( the_schema [i].id_run[j] );
      /* In case a duplicate ID has been found */
      if ( uniqueRunnerIDs.size() == nbUniqueIDs ) {
        the_schema [i].execution_id_run[j] = ++maxSpecifiedRunnerID;
      }
      nbUniqueIDs = uniqueRunnerIDs.size();
    }
  }

  /* Looking for my node */
  for (unsigned i = 0; i < the_schema.size (); i ++) {
    if (the_schema [i].rk == getNodeRank ())
      my_node = & (the_schema [i]);
  }


  /* About me */
  char mess [1000];

  sprintf (mess, "my rank is %d", my_node -> rk);
  printDebugMessage (mess);

  if (isScheduleNode ())
    printDebugMessage ("I'am a scheduler");  

  for (unsigned i = 0; i < my_node -> id_run.size (); i ++) {
    sprintf (mess, "I manage the runner %d", my_node -> id_run [i]);
    printDebugMessage (mess);
  }

  if (my_node -> num_workers) {

    sprintf (mess, "I manage %d worker(s)", my_node -> num_workers);
    printDebugMessage (mess);
  }

  closeXMLDocument ();
}
