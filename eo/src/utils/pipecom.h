/* ----------------------------------------------------------------------
 * Where........: CMAP - Polytechnique 
 * File.........: pipecom.h
 * Author.......: Bertrand Lamy (EEAAX)
 * Created......: Thu Mar  9 17:21:15 1995
 * Description..: Pipe communication with a process
 * 
 * Ident........: $Id: pipecom.h,v 1.3 2001-10-08 09:13:16 evomarc Exp $
 * ----------------------------------------------------------------------
 */

#ifndef PIPECOM_H
#define PIPECOM_H

// this file cannot be used from C or C++ any more due to some const additions
// however, if you remove the const, it should work in C also
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

int PipeComSend( PCom *to, const char *line );
int PipeComSendn( PCom *to, const char *data, int n );

int PipeComReceive( PCom *from, char *data, int max );

int PipeComClose( PCom *to );

int PipeComWaitFor( PCom *from, char *what );

#ifdef __cplusplus
} /* ferme extern "C" */
#endif

#endif
