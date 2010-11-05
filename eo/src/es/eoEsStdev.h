/* (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000

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


#ifndef _eoEsStdev_h
#define _eoEsStdev_h

#include <eoVector.h>


/** Evolutionary Strategy with a standard deviation per parameter

Evolutionary strategie style representation, supporting co-evolving
standard deviations.

@ingroup Real
*/
template <class Fit>
class eoEsStdev : public eoVector<Fit, double>
{
public:

    using eoVector<Fit, double>::size;

    typedef double Type;

    eoEsStdev(void) : eoVector<Fit, double>() {}

    virtual std::string className(void) const { return "eoEsStdev"; }

    void printOn(std::ostream& os) const
    {
        eoVector<Fit,double>::printOn(os);
        os << ' ';
        std::copy(stdevs.begin(), stdevs.end(), std::ostream_iterator<double>(os, " "));
        os << ' ';
    }

    void readFrom(std::istream& is)
    {
        eoVector<Fit,double>::readFrom(is);
        stdevs.resize(size());
        unsigned i;
        for (i = 0; i < size(); ++i)
            is >> stdevs[i];
    }


    std::vector<double> stdevs;
};

#endif



// Local Variables:
// coding: iso-8859-1
// mode:C++
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// End:
