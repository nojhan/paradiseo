// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "schema.cpp"

// (c) OPAC Team, LIFL, August 2005

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

#include <iostream>
#include <assert.h>

#include "schema.h"
#include "xml_parser.h"
#include "comm.h"
#include "node.h"
#include "../../core/peo_debug.h"

std :: vector <Node> the_schema;

Node * my_node;

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
      /* TAG: </runner> */
      assert (getNextNode () == "runner");
    }
    else {      
      /* TAG: </node> */
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

  /* Looking for my node */
  for (unsigned i = 0; i < the_schema.size (); i ++)
    if (the_schema [i].rk == getNodeRank ())
      my_node = & (the_schema [i]);
  
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

