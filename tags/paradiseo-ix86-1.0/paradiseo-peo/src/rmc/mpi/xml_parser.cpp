// "xml_parser.h"

// (c) OPAC Team, LIFL, August 2005

/* 
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

