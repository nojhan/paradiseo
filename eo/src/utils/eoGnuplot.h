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
#ifndef EO_eoGnuplot_H
#define EO_eoGnuplot_H

#include <string>

#include "pipecom.h"


/** Base class for calls to gnuplot

This class is the abstract class that will be used by further gnuplot
calls to plots what is already written by some eoMonitor into a file

@author Marc Schoenauer
@version 0.0 (2001)

@ingroup Monitors
*/
class eoGnuplot
{
public:

    /** Open pipe to Gnuplot.

    @param _title Title for gnuplot window.
    @param _extra Extra parameters to gnuplot (default to none: "").
    */
    eoGnuplot(std::string _title, std::string _extra = std::string(""));

    /** Destructor

    Close the gnuplot windows if pipe was correctly opened
    */
    virtual ~eoGnuplot();

    /** Class name */
    virtual std::string className() const
        { return "eoGnuplot"; }

    /** Send command to gnuplot */
    void gnuplotCommand(const char * _command);

    /** Send command to gnuplot

    @overload
    */
    void gnuplotCommand(std::string _command)
        { gnuplotCommand(_command.c_str()); }


protected:

    /** Initialize gnuplot

    @param _title Title for gnuplot window.
    @param _extra Extra parameters to gnuplot.
    */
    void initGnuPlot(std::string _title, std::string _extra);

    /** The stats might be unknown in Ctor */
    bool firstTime;

    /** Communication with gnuplot OK */
    PCom *gpCom;

    /** Internal counter for gnuplot windows */
    static unsigned numWindow;
};


#endif // EO_eoGnuplot_H



// Local Variables:
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// mode: C++
// End:
