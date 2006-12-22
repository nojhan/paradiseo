// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "runner.cpp"

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

#include "../../core/messaging.h"
#include "../../core/runner.h"
#include "node.h"
#include "send.h"
#include "tags.h"
#include "schema.h"

bool Runner :: isLocal () {

  for (unsigned i = 0; i < my_node -> id_run.size (); i ++)
    if (my_node -> id_run [i] == id)
      return true;
  return false;
}

void Runner :: packTermination () {

  pack (id);
}

void Runner :: terminate () {

  sendToAll (this, RUNNER_STOP_TAG);     
}

