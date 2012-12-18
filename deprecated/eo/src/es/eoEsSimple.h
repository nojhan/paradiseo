/* (c) GeNeura Team, 2000 - EEAAX 1999, Maarten Keijzer 2000

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place, Suite 330, Boston, MA 02111-1307 USA

Contact: http://eodev.sourceforge.net
         todos@geneura.ugr.es, http://geneura.ugr.es
         Marc.Schoenauer@polytechnique.fr
         mak@dhi.dk
 */


#ifndef _eoEsSimple_h
#define _eoEsSimple_h

#include <EO.h>
#include <vector>

#include <eoVector.h>

/** Simple Evolution Strategy

@ingroup Real

One of the more simple evolution strategies, sporting just a single stdeviation
for the entire chromosome. For more advanced versions see also eoEsStdev
eoEsFull

@see eoEsStdev eoEsFull
*/
template <class Fit>
class eoEsSimple : public eoVector<Fit, double>
{
public :

    typedef double Type;

    eoEsSimple() : eoVector<Fit, double>() {}

    virtual std::string className() const { return "eoEsSimple"; }

    void printOn(std::ostream& os) const
    {
        eoVector<Fit,double>::printOn(os);
        os << ' ' << stdev << ' ';
    }

    void readFrom(std::istream& is)
    {
        eoVector<Fit,double>::readFrom(is);
        is >> stdev;
    }

    double stdev;
};

#endif



// Local Variables:
// coding: iso-8859-1
// mode:C++
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// End:
