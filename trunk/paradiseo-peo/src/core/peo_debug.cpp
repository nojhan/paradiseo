// "peo_debug.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "peo_debug.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "peo_debug.h"

#define MAX_BUFF_SIZE 1000

#define DEBUG_PATH "./log/"

static bool debug = true;

static char host [MAX_BUFF_SIZE];

std :: vector <FILE *> files;

void setDebugMode (bool __dbg) {

  debug = __dbg;
  gethostname (host, MAX_BUFF_SIZE);
}

extern int getNodeRank ();

void initDebugging () {
  
  mkdir (DEBUG_PATH, S_IRWXU);
  //  files.push_back (stdout);
  char buff [MAX_BUFF_SIZE];
  sprintf (buff, "%s/%d", DEBUG_PATH, getNodeRank ());
  files.push_back (fopen (buff, "w"));
}

void endDebugging () {

  for (unsigned i = 0; i < files.size (); i ++)
    if (files [i] != stdout)
      fclose (files [i]);
}

void printDebugMessage (const char * __mess) {

  if (debug) {

    char buff [MAX_BUFF_SIZE];
    time_t t = time (0);

    /* Date */
    sprintf (buff, "[%s][%s: ", host, ctime (& t));
    * strchr (buff, '\n') = ']';
    for (unsigned i = 0; i < files.size (); i ++)
      fprintf (files [i], buff);

    /* Message */
    sprintf (buff, "%s", __mess);
    
    for (unsigned i = 0; i < files.size (); i ++) {
      fputs (buff, files [i]);
      fputs ("\n", files [i]);
      fflush (files [i]);
    }
  }
}
