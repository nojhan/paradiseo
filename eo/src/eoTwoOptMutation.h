//-----------------------------------------------------------------------------
// eoTwoOptMutation.h
// (c) GeNeura Team, 2000 - EEAAX 2000 - Maarten Keijzer 2000
// (c) INRIA Futurs - Dolphin Team - Thomas Legrand 2007
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
thomas.legrand@lifl.fr
Marc.Schoenauer@polytechnique.fr
mak@dhi.dk
*/
//-----------------------------------------------------------------------------

#ifndef eoTwoOptMutation_h
#define eoTwoOptMutation_h

//-----------------------------------------------------------------------------


/**
* Especially designed for combinatorial problem such as the TSP.
*
* @ingroup Variators
*/
template<class EOT> class eoTwoOptMutation: public eoMonOp<EOT>
{
public:

    typedef typename EOT::AtomType GeneType;

    /// CTor
    eoTwoOptMutation(){}

    /// The class name.
    virtual std::string className() const { return "eoTwoOptMutation"; }

    /**
    *
    * @param _eo The cromosome which is going to be changed.
    */
    bool operator()(EOT& _eo) {
        // generate two different indices
        unsigned i(eo::rng.random(_eo.size()));
        unsigned j;
        do {
            j = eo::rng.random(_eo.size());
        } while(i == j);
        unsigned from(std::min(i,j));
        unsigned to(std::max(i,j));
        unsigned idx((to - from)/2);

        // inverse between from and to
        for(unsigned k = 0; k <= idx; ++k)
            std::swap(_eo[from+k],_eo[to-k]);
        return true;
    }

};
/** @example t-eoTwoOptMutation.cpp
 */


//-----------------------------------------------------------------------------
#endif


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
