/* ----------------------------------------------------------------------
 * Where........: CMAP - Polytechnique
 * File.........: pipecom.c
 * Author.......: Bertrand Lamy (Equipe genetique)
 * Created......: Mon Mar 13 13:50:11 1995
 * Description..: Communication par pipe bidirectionnel avec un autre process
 * ----------------------------------------------------------------------
 */

// MSC equivalent must be written and tested or some #idef instructions added
// with a clear message at compile time that this is for Unix only ???

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _WINDOWS

#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "pipecom.h"



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


int PipeComSend( PCom *to, const char *line )
{
    int	nb = 0;
    if( ! Check(to ) )
        return nb;
    nb = fprintf( to->fWrit, line, 0 );
    fflush( to->fWrit );
    return nb;
}


int PipeComSendn( PCom *to, const char *data, int n )
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

#endif /*_MSC_VER*/


// Local Variables:
// coding: iso-8859-1
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
