// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGnuplot1DMonitor.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
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

   Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoGnuplot1DMonitor_H
#define _eoGnuplot1DMonitor_H

#include <string>

#include <utils/eoMonitor.h>
#include <eoObject.h>

/**
@author Marc Schoenauer 2000
@version 0.0

This class plots through gnuplot the eoStat given as argument

*/
//-----------------------------------------------------------------------------

#include <fstream>
// #include "pipecom.h"
// 
// this is pipecom.h

/* ----------------------------------------------------------------------
 * Where........: CMAP - Polytechnique 
 * File.........: pipecom.h
 * Author.......: Bertrand Lamy (EEAAX)
 * Created......: Thu Mar  9 17:21:15 1995
 * Description..: Pipe communication with a process
 * 
 * Ident........: $Id: eoGnuplot1DMonitor.h,v 1.2 2000-11-29 17:20:16 evomarc Exp $
 * ----------------------------------------------------------------------
 */

#ifndef PIPECOM_H
#define PIPECOM_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>


typedef struct PipeCommunication { 
    FILE	*fWrit;
    FILE	*fRead;
    int		pid;
} PCom;
    


PCom *PipeComOpen( char *prog );
PCom *PipeComOpenArgv( char *prog, char *argv[] );

int PipeComSend( PCom *to, char *line );
int PipeComSendn( PCom *to, char *data, int n );

int PipeComReceive( PCom *from, char *data, int max );

int PipeComClose( PCom *to );

int PipeComWaitFor( PCom *from, char *what );

#ifdef __cplusplus
} /* ferme extern "C" */
#endif

#endif	/* 		PIPECOM_H */



/** eoGnuplot1DMonitor plots stats through gnuplot
 */
class eoGnuplot1DMonitor: public eoFileMonitor 
{
 public:
    // Ctor
  eoGnuplot1DMonitor(std::string _filename) : 
      eoFileMonitor(_filename, " "), firstTime(true)
  {
    // opens pipe with Gnuplot
    initGnuPlot(_filename);
  }
  
  // Dtor
  virtual ~eoGnuplot1DMonitor() {
  // close - the gnuplot windows if pipe was correctly opened
    if( gpCom ) {
      PipeComSend( gpCom, "quit\n" );	// Ferme gnuplot
      PipeComClose( gpCom );
      gpCom =NULL;
    }
  }

  virtual eoMonitor&  operator() (void) ;

  /// Class name.
  virtual string className() const { return "eoGnuplot1DMonitor"; }

private: 
  void initGnuPlot(std::string _filename);
  void  FirstPlot();  
  // the private data
  bool firstTime;       // the stats might be unknown in Ctor
  PCom        *gpCom;	  // Communication with gnuplot OK
};

// the following should be placed in a separate eoGnuplot1DMonitor.cpp 

////////////////////////////////////////////////////////////
eoMonitor&   eoGnuplot1DMonitor::operator() (void)
  /////////////////////////////////////////////////////////
{
  // update file using the eoFileMonitor
  eoFileMonitor::operator()();

  // sends plot order to gnuplot
  // assumes successive plots will have same nb of columns!!!
  if (firstTime)
    {
      FirstPlot();
      firstTime = false;
    }
  else 
    {
      if( gpCom ) {
	PipeComSend( gpCom, "replot\n" );	
      }
    }
  return *this;
}

////////////////////////////////////////////////////////////
void eoGnuplot1DMonitor::initGnuPlot(std::string _filename) 
  /////////////////////////////////////////////////////////
{
  char	*args[6];
  args[0] = strdup( "gnuplot" );
  args[1] = strdup( "-geometry" );
  args[2] = strdup( "300x200-0+0" );
  args[3] = strdup( "-title" );
  args[4] = strdup( _filename.c_str() ); 
  args[5] = 0;
  gpCom = PipeComOpenArgv( "gnuplot", args );
  if( ! gpCom )
    throw runtime_error("Impossible to spawn gnuplot\n");
  else {
    PipeComSend( gpCom, "set grid\n" );
    PipeComSend( gpCom, "set data style lines\n" );
  }
}

    
////////////////////////////////////////////////////////////
void  eoGnuplot1DMonitor::FirstPlot() 
  ////////////////////////////////////////////////////////
{
  if (vec.size() < 2) 
    {
      throw runtime_error("Must have some stats to plot!\n");
    }
  char buff[1024];
  ostrstream os(buff, 1024);
  os << "plot";
  for (unsigned i=1; i<vec.size(); i++) {
    os << " '" << getFileName().c_str() <<
      "' using 1:" << i+1 << " title '" << vec[i]->longName() << "'" ;
    if (i<vec.size()-1)
      os << ", ";
  }
  os << "\n";
  os << '\0';
  PipeComSend( gpCom, buff );
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
 * Ident........: $Id: eoGnuplot1DMonitor.h,v 1.2 2000-11-29 17:20:16 evomarc Exp $
 * ----------------------------------------------------------------------
 */


#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

// #include "pipecom.h"


int Check( PCom *com )
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


PCom * PipeComOpen( char *prog )
{
    char	*args[2];
    args[0] = prog;
    args[1] = NULL;
    return PipeComOpenArgv( prog, args ); 
}


PCom * PipeComOpenArgv( char *prog, char *argv[] )
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


int PipeComSend( PCom *to, char *line )
{
    int	nb = 0;
    if( ! Check(to ) )
	return nb;
    nb = fprintf( to->fWrit, line );
    fflush( to->fWrit );
    return nb;
}


int PipeComSendn( PCom *to, char *data, int n )
{
    int	nb = 0;
    if( ! Check(to) ) 
	return nb;

    nb = fwrite( data, 1, n, to->fWrit );
    fflush( to->fWrit );
    return nb;
}


int PipeComReceive( PCom *from, char *data, int max )
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



int PipeComClose( PCom *to )
{
    if( ! Check(to) )
	return 0;
    fclose( to->fRead );
    fclose( to->fWrit );
    free( to );
    return 1;
}



int PipeComWaitFor( PCom *from, char *what )
{
    char	buffer[256];
    do {
	if( ! PipeComReceive( from, buffer, 256 ) )
	    return 0;
    } while( strcmp( buffer, what ) );
    return 1;
}


#endif _eoGnuplot1DMonitor_H
