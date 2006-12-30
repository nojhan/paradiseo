// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "mess.h"

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

#ifndef __mess_rmc_h
#define __mess_rmc_h

#include "../../core/messaging.h"

extern void initMessage ();

extern void sendMessage (int __to, int __tag);

extern void sendMessageToAll (int __tag);

extern void receiveMessage (int __from, int __tag);

extern void cleanBuffers ();

extern void waitBuffers ();

extern bool probeMessage (int & __src, int & __tag);

extern void waitMessage ();

#endif

