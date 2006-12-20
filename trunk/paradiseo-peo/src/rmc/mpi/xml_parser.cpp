// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "xml_parser.h"

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <libxml/xmlreader.h>

#include "xml_parser.h"

static xmlTextReaderPtr reader;

void openXMLDocument (const char * __filename) {
  
  reader = xmlNewTextReaderFilename (__filename);
  
  if (! reader) {
    
    fprintf (stderr, "unable to open '%s'.\n", __filename);
    exit (1);
  }
}

void closeXMLDocument () {

  xmlFreeTextReader (reader);
}

std :: string getAttributeValue (const std :: string & __attr) {
  
  xmlChar * value = xmlTextReaderGetAttribute (reader, (const xmlChar *) __attr.c_str ());
  
  std :: string str ((const char *) value);
  
  xmlFree (value);
  
  return str;
}

static bool isSep (const xmlChar * __text) {
  
  for (unsigned i = 0; i < strlen ((char *) __text); i ++)
    if (__text [i] != ' ' && __text [i] != '\t' && __text [i] != '\n')
      return false;
  return true;
}

std :: string getNextNode () {
  
  xmlChar * name, * value;

  do {
    xmlTextReaderRead (reader);
    name = xmlTextReaderName (reader);
    value = xmlTextReaderValue (reader);
    //    printf ("value = %s\n", value);
  } while (! strcmp ((char *) name, "#text") && isSep (value));

  std :: string str;

  if (strcmp ((char *) name, "#text"))
    str.assign ((char *) name);
  else
    str.assign ((char *) value);
  
  if (name)
    xmlFree (name);
  if (value)
    xmlFree (value);
    
  return str;
}

