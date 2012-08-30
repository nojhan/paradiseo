//-----------------------------------------------------------------------------
// (c) Marc Schoenauer, 2001
// Copyright (C) 2005 Jochen Küpper
/*
   This library is free software; you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the Free
   Software Foundation; either version 2 of the License, or (at your option) any
   later version.

   This library is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
   details.

   You should have received a copy of the GNU Lesser General Public License
   along with this library; if not, write to the Free Software Foundation, Inc.,
   59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

   Contact: Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <sstream>
#include <stdexcept>

#include "eoGnuplot.h"


unsigned eoGnuplot::numWindow=0;



eoGnuplot::eoGnuplot(std::string _title, std::string _extra)
    : firstTime(true)
{
    initGnuPlot(_title, _extra);
}



eoGnuplot::~eoGnuplot()
{
#ifdef HAVE_GNUPLOT
    if( gpCom ) {
        PipeComSend( gpCom, "quit\n" );
        PipeComClose( gpCom );
        gpCom =NULL;
    }
#endif
}



void eoGnuplot::gnuplotCommand(const char *_command)
{
#ifdef HAVE_GNUPLOT
    if(gpCom) {
        PipeComSend( gpCom, _command );
        PipeComSend( gpCom, "\n" );
    }
#endif
}



void eoGnuplot::initGnuPlot(std::string _title, std::string _extra)
{
#ifdef HAVE_GNUPLOT
    std::ostringstream os;
    os << "250x150-0+" << 170 * numWindow++;
    char *args[6];
    args[0] = strdup( GNUPLOT_PROGRAM );
    args[1] = strdup( "-geometry" );
    args[2] = strdup( os.str().c_str());
    args[3] = strdup( "-title" );
    args[4] = strdup( _title.c_str() );
    args[5] = 0;
    gpCom = PipeComOpenArgv( GNUPLOT_PROGRAM, args );
    if(! gpCom )
        throw std::runtime_error("Cannot spawn gnuplot\n");
    else {
        PipeComSend( gpCom, "set grid\n" );
        PipeComSend( gpCom, _extra.c_str() );
        PipeComSend( gpCom, "\n" );
    }
#endif
}



// Local Variables:
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
