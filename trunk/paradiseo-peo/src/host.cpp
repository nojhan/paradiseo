// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "host.cpp"

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

#include "host.h"

#define MAX_BUFF_SIZE 1000

std :: string getShortHostName (const std :: string & __host_name) {

  * strchr (__host_name.c_str (), '.') = '\0';
  
  return __host_name;
}

Host :: Host (const std :: string & __name,
	      unsigned __num_procs
	      ) : name (__name),
		  num_procs (__num_procs) {  
  
  //  printf ("yeahhhh\n");
}

void Host :: save (FILE * __f) {
  
  fprintf (__f, "\t\t<host name=\"%s\" num_procs=\"%u\"/>\n", name.c_str (), num_procs);
}

std :: string getDomainName (const std :: string & __host_name) {

  return strchr (__host_name.c_str (), '.');
}
