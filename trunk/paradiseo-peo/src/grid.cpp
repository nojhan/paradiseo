// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "grid.cpp"

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
#include <fstream>

#include <libxml/xmlreader.h>

#include "grid.h"

std :: vector <Cluster> the_grid;

void saveGrid (const char * __filename) {

  FILE * f = fopen (__filename, "w");

  fprintf (f, "<?xml version=\"1.0\"?>\n\n");
  
  fprintf (f, "<grid name=\"grid5000\">\n"); 
  for (unsigned i = 0; i < the_grid.size (); i ++)
    the_grid [i].save (f);  
  fprintf (f, "</grid>\n");
  fclose (f);
}

static void processNode(xmlTextReaderPtr reader) {
    
  const xmlChar * name, * value;

  name = xmlTextReaderConstName(reader);
  if (name == NULL)
    name = BAD_CAST "--";
  printf ("#%s#\n", name);
  value = xmlTextReaderConstValue(reader);    
  if (value && ! strcmp ((const char *) name, "name"))
      printf("%s\n", value);
}

void loadGrid (const char * __filename) {

  xmlTextReaderPtr reader = xmlReaderForFile (__filename, NULL, 0);
  
  int ret = xmlTextReaderRead (reader);
  while (ret == 1) {
    processNode (reader);
    ret = xmlTextReaderRead(reader);
  }
  xmlFreeTextReader(reader);
}

Cluster * getCluster (const std :: string & __domain_name) {
  
  for (unsigned i = 0; i < the_grid.size (); i ++)    
    if (the_grid [i].domain_name == __domain_name)      
      return & the_grid [i];
  return 0;
}

void loadMachineFile (const char * __filename) {

  std :: ifstream f (__filename);

  while (true) {

    std :: string host_name;
    f >> host_name;
    
    if (f.eof ())
      break;
    std :: string domain_name = getDomainName (host_name);
    std :: string short_host_name = getShortHostName (host_name);
    Cluster * cluster = getCluster (domain_name);    
    Host host (short_host_name);
    if (cluster)
      cluster -> push_back (host);
    else {
      the_grid.push_back (Cluster (domain_name));
      the_grid.back ().push_back (Host (short_host_name));
    }            
  }
  
  f.close ();
}
