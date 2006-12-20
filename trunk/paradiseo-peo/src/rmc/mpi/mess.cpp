// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "mess.cpp"

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

#include <mpi.h>
#include <vector>

#include "mess.h"
#include "../../core/peo_debug.h"
#include "node.h"

#define MPI_BUF_SIZE 1024*64
	
static char mpi_buf [MPI_BUF_SIZE];
	
static int pos_buf ;

static std :: vector <char *> act_buf; /* Active buffers */

static std :: vector <MPI_Request *> act_req; /* Active requests */

void cleanBuffers () {

  for (unsigned i = 0; i < act_req.size ();) {
       
    MPI_Status stat ;
    int flag ;
    MPI_Test (act_req [i], & flag, & stat) ;
    if (flag) {
      
      delete act_buf [i] ;
      delete act_req [i] ;
	
      act_buf [i] = act_buf.back () ;
      act_buf.pop_back () ;
      
      act_req [i] = act_req.back () ;
      act_req.pop_back () ;
    }
    else
      i ++;
  } 
}

void waitBuffers () {

  printDebugMessage ("waiting the termination of the asynchronous operations to complete");

  for (unsigned i = 0; i < act_req.size (); i ++) {
       
    MPI_Status stat ;

    MPI_Wait (act_req [i], & stat) ;
      
    delete act_buf [i] ;
    delete act_req [i] ;
  } 
}

bool probeMessage (int & __src, int & __tag) {

  int flag;

  MPI_Status stat;

  MPI_Iprobe (MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, & flag, & stat);

  __src = stat.MPI_SOURCE;
  __tag = stat.MPI_TAG;

  return flag;
}

void waitMessage () {

  MPI_Status stat;  

  MPI_Probe (MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, & stat);
}

void initMessage () {
  
  pos_buf = 0;
}

void sendMessage (int __to, int __tag) {

  cleanBuffers ();  
  act_buf.push_back (new char [pos_buf]);
  act_req.push_back (new MPI_Request);  
  memcpy (act_buf.back (), mpi_buf, pos_buf);  
  MPI_Isend (act_buf.back (), pos_buf, MPI_PACKED, __to, __tag, MPI_COMM_WORLD, act_req.back ()); 
}

void sendMessageToAll (int __tag) {

  for (int i = 0; i < getNumberOfNodes (); i ++)
    sendMessage (i, __tag);
}

void receiveMessage (int __from, int __tag) {
  
  MPI_Status stat;  
  MPI_Request req;

  MPI_Irecv (mpi_buf, MPI_BUF_SIZE, MPI_PACKED, __from, __tag, MPI_COMM_WORLD, & req) ;
  MPI_Wait (& req, & stat) ;
}

/* Char */
void pack (const char & __c) {

  MPI_Pack ((void *) & __c, 1, MPI_CHAR, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Float */
void pack (const float & __f, int __nitem) {

  MPI_Pack ((void *) & __f, __nitem, MPI_FLOAT, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Double */
void pack (const double & __d, int __nitem) {

  MPI_Pack ((void *) & __d, __nitem, MPI_DOUBLE, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Integer */
void pack (const int & __i, int __nitem) {

  MPI_Pack ((void *) & __i, __nitem, MPI_INT, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Unsigned int. */
void pack (const unsigned int & __ui, int __nitem) {

  MPI_Pack ((void *) & __ui, __nitem, MPI_UNSIGNED, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Short int. */
void pack (const short & __sh, int __nitem) {

  MPI_Pack ((void *) & __sh, __nitem, MPI_SHORT, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Unsigned short */
void pack (const unsigned short & __ush, int __nitem) {

  MPI_Pack ((void *) & __ush, __nitem, MPI_UNSIGNED_SHORT, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Long */
void pack (const long & __l, int __nitem) {

  MPI_Pack ((void *) & __l, __nitem, MPI_LONG, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Unsigned long */
void pack (const unsigned long & __ul, int __nitem) {

  MPI_Pack ((void *) & __ul, __nitem, MPI_UNSIGNED_LONG, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* String */
void pack (const char * __str) {
  
  int len = strlen (__str) + 1;
  MPI_Pack (& len, 1, MPI_INT, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
  MPI_Pack ((void *) __str, len, MPI_CHAR, mpi_buf, MPI_BUF_SIZE, & pos_buf, MPI_COMM_WORLD);
}

/* Char */
void unpack (char & __c) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __c, 1, MPI_CHAR, MPI_COMM_WORLD);
}

/* Float */
void unpack (float & __f, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __f, __nitem, MPI_FLOAT, MPI_COMM_WORLD);
}

/* Double */
void unpack (double & __d, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __d, __nitem, MPI_DOUBLE, MPI_COMM_WORLD);
}

/* Integer */
void unpack (int & __i, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __i, __nitem, MPI_INT, MPI_COMM_WORLD);
}

/* Unsigned int. */
void unpack (unsigned int & __ui, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __ui, __nitem, MPI_UNSIGNED, MPI_COMM_WORLD);
}

/* Short int. */
void unpack (short & __sh, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __sh, __nitem, MPI_SHORT, MPI_COMM_WORLD);
}

/* Unsigned short */
void unpack (unsigned short & __ush, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __ush, __nitem, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD);
}

/* Long */
void unpack (long & __l, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __l, __nitem, MPI_LONG, MPI_COMM_WORLD);
}

/* Unsigned long */
void unpack (unsigned long & __ul, int __nitem) {

  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & __ul, __nitem, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
}

/* String */
void unpack (char * __str) {

  int len;
  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, & len, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Unpack (mpi_buf, MPI_BUF_SIZE, & pos_buf, __str, len, MPI_CHAR, MPI_COMM_WORLD);    
}

