// "xml_parser.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __xml_parser_h
#define __xml_parser_h

#include <string>

extern void openXMLDocument (const char * __filename);

extern void closeXMLDocument ();

extern std :: string getAttributeValue (const std :: string & __attr);

extern std :: string getNextNode ();

#endif
