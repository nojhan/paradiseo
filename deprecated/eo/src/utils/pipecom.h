/* ----------------------------------------------------------------------
 * Where........: CMAP - Polytechnique
 * File.........: pipecom.h
 * Author.......: Bertrand Lamy (EEAAX)
 * Created......: Thu Mar  9 17:21:15 1995
 * Description..: Pipe communication with a process
 * ----------------------------------------------------------------------
 */

// This file cannot be used from C any more due to some const additions.
// However, if you remove the const, it should work in C as well.

#ifndef EO_PIPECOM_H
#define EO_PIPECOM_H

#include <stdio.h>


typedef struct PipeCommunication {
    FILE	*fWrit;
    FILE	*fRead;
    int		pid;
} PCom;


extern PCom *PipeComOpen( char *prog );
extern PCom *PipeComOpenArgv( char *prog, char *argv[] );

extern int PipeComSend( PCom *to, const char *line );
extern int PipeComSendn( PCom *to, const char *data, int n );

extern int PipeComReceive( PCom *from, char *data, int max );

extern int PipeComClose( PCom *to );
extern int PipeComWaitFor( PCom *from, char *what );


#endif // EO_PIPECOM_H



// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
