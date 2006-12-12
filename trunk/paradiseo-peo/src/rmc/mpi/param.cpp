// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "param.cpp"

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

#include <utils/eoParser.h>

#include "schema.h"

void loadRMCParameters (int & __argc, char * * & __argv) {

  eoParser parser (__argc, __argv);

  /* Schema */
  eoValueParam <std :: string> schema_param ("schema.xml", "schema", "?");
  parser.processParam (schema_param);
  loadSchema (schema_param.value ().c_str ());
}
