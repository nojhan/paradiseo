// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGnuplot1DMonitor.h
// (c) Marc Schoenauer, 2001
/* 
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

   Contact: Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoGnuplot_H
#define _eoGnuplot_H

#include <string>

/**
@author Marc Schoenauer 2001
@version 0.0

This class is the abstract class that will be used by further gnuplot calls
to plots what is already written by some eoMonitor into a file

*/
//-----------------------------------------------------------------------------

#include <fstream>
#include <utils/pipecom.h>


/** eoGnuplot: base class that will be used to call gnuplot
 */
class eoGnuplot
{
 public:
    // Ctor
  eoGnuplot(std::string _title, std::string _extra = string("")) : 
    firstTime(true)
  {
    // opens pipe with Gnuplot
    initGnuPlot(_title, _extra);
  }
  
  // Dtor
  virtual ~eoGnuplot() {
  // close - the gnuplot windows if pipe was correctly opened
    if( gpCom ) {
      PipeComSend( gpCom, "quit\n" );	// Ferme gnuplot
      PipeComClose( gpCom );
      gpCom =NULL;
    }
  }

  /// Class name.
  virtual string className() const { return "eoGnuplot"; }

  /** send a command to gnuplot directly
   */
  void gnuplotCommand(std::string _command)
  {
    if( gpCom ) {
      PipeComSend( gpCom, _command.c_str() );
      PipeComSend( gpCom, "\n" );
    }
  }


protected:
  void initGnuPlot(std::string _title, std::string _extra);
  // the private data
  bool firstTime;       // the stats might be unknown in Ctor
  PCom        *gpCom;	  // Communication with gnuplot OK
private:
};

// the following should be placed in a separate eoGnuplot.cpp

  static unsigned numWindow=0;

////////////////////////////////////////////////////////////
inline void eoGnuplot::initGnuPlot(std::string _title, std::string _extra)
  /////////////////////////////////////////////////////////
{
  char snum[255];
  ostrstream os(snum, 254);
  os << "300x200-0+" << numWindow*220 << ends;
  numWindow++;
  char	*args[6];
  args[0] = strdup( "gnuplot" );
  args[1] = strdup( "-geometry" );
  args[2] = strdup( os.str() );
  args[3] = strdup( "-title" );
  args[4] = strdup( _title.c_str() );
  args[5] = 0;
  gpCom = PipeComOpenArgv( "gnuplot", args );
  if( ! gpCom )
    throw runtime_error("Impossible to spawn gnuplot\n");
  else {
    PipeComSend( gpCom, "set grid\n" );
    PipeComSend( gpCom, _extra.c_str() );
    PipeComSend( gpCom, "\n" );
  }
}


// the following should be placed in a separate file pipecom.c
// together with the corresponding pipecom.h
// but first their MSC equivalent must be written and tested
// or some #idef instructions put with clear message at compile time
// that this is for Unix only ???

/* ----------------------------------------------------------------------
 * Where........: CMAP - Polytechnique
 * File.........: pipecom.c
 * Author.......: Bertrand Lamy (Equipe genetique)
 * Created......: Mon Mar 13 13:50:11 1995
 * Description..: Communication par pipe bidirectionnel avec un autre process
 *
 * Ident........: $Id: eoGnuplot.h,v 1.3 2001-02-12 13:58:51 maartenkeijzer Exp $
 * ----------------------------------------------------------------------
 */


#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

// #include "pipecom.h"


inline int Check( PCom *com )
{
    if( ! com ) {
	fprintf( stderr, "PipeCom: Null pointer.\n" );
	fflush( stderr );
	return 0;
    }
    if( kill( com->pid, 0 ) != 0 ) {
	fprintf( stderr, "PipeCom: process doesn't exists.\n" );
	fflush( stderr );
	return 0;
    }
    return 1;
}


inline PCom * PipeComOpen( char *prog )
{
    char	*args[2];
    args[0] = prog;
    args[1] = NULL;
    return PipeComOpenArgv( prog, args );
}


inline PCom * PipeComOpenArgv( char *prog, char *argv[] )
{
    int		toFils[2];
    int		toPere[2];
    int		sonPid;
    PCom	* ret = NULL;

    if( pipe( toFils ) < 0 ) {
	perror( "PipeComOpen: Creating pipes" );
	return ret;
    }
    if( pipe( toPere ) < 0 ) {
	perror( "PipeComOpen: Creating pipes" );
	return ret;
    }

    switch( (sonPid = vfork()) ) {
    case -1:
	perror("PipeComOpen: fork failed" );
	return ret;
	break;

    case 0:
	/* --- Here's the son --- */
	/* --- replace old stdin --- */
	if( dup2( toFils[0], fileno(stdin) ) < 0 ) {
	    perror( "PipeComOpen(son): could not connect" );
	    exit( -1 );
	    /* --- AVOIR: kill my father --- */
	}
	if( dup2( toPere[1], fileno(stdout) ) < 0 ) {
	    perror( "PipeComOpen(son): could not connect" );
	    exit( -1 );
	}
	if( execvp( prog, argv ) < 0 ) {
	    perror( prog );
	    perror( "PipeComOpen: can't exec" );
	    exit(1);
	}
	break;
    default:
	ret = (PCom *) malloc( sizeof(PCom) );
	if( ! ret )
	    return NULL;

	ret->fWrit = (FILE *)fdopen( toFils[1], "w" );
	ret->fRead = (FILE *)fdopen( toPere[0], "r" );
	ret->pid = sonPid;
    }
    return ret;
}


inline int PipeComSend( PCom *to, const char *line )
{
    int	nb = 0;
    if( ! Check(to ) )
	return nb;
    nb = fprintf( to->fWrit, line );
    fflush( to->fWrit );
    return nb;
}


inline int PipeComSendn( PCom *to, const char *data, int n )
{
    int	nb = 0;
    if( ! Check(to) )
	return nb;

    nb = fwrite( data, 1, n, to->fWrit );
    fflush( to->fWrit );
    return nb;
}


inline int PipeComReceive( PCom *from, char *data, int max )
{
    if( ! Check(from) )
	return 0;
    if( ! data ) {
      fprintf( stderr, "PipeComReceive: Invalid data pointer\n" );
      fflush( stderr );
      return 0;
    }
    if( fgets( data, max, from->fRead ) )
	return strlen(data);
    return 0;
}



inline int PipeComClose( PCom *to )
{
    if( ! Check(to) )
	return 0;
    fclose( to->fRead );
    fclose( to->fWrit );
    free( to );
    return 1;
}



inline int PipeComWaitFor( PCom *from, char *what )
{
    char	buffer[256];
    do {
	if( ! PipeComReceive( from, buffer, 256 ) )
	    return 0;
    } while( strcmp( buffer, what ) );
    return 1;
}


#endif _eoGnuplot_H
