// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsObjectiveBounds.h
// (c) Maarten Keijzer 2000, GeNeura Team, 1998 - EEAAX 1999
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoEsObjectiveBounds_h
#define _eoEsObjectiveBounds_h

/**
\defgroup EvolutionStrategies

  Various classes for the initialization and mutation of real valued vectors.

  Supports simple mutations and various more adaptable mutations, including
  correlated mutations.

*/


/**
\class eoEsObjectiveBounds eoEsObjectiveBounds.h es/eoEsObjectiveBounds.h
\ingroup EvolutionStrategies

    Defines the minima and maxima of the object variables. Needed by eoEsChromInit
    and eoEsMutate

    @see eoEsChromInit eoEsMutate
*/
class eoEsObjectiveBounds
{
    public :
        
        /** 
        Objective bounds for a global minimum and maximum
        */
        eoEsObjectiveBounds(int _nGenes, double _min, double _max) : repMinimum(_nGenes), repMaximum(_nGenes)
        {
            std::fill(repMinimum.begin(), repMinimum.end(), _min);
            std::fill(repMaximum.begin(), repMaximum.end(), _max);
        }

        /** 
        Objective bounds for a per gene minimum and maximum
        */
        eoEsObjectiveBounds(const std::vector<double>& _min, const std::vector<double>& _max)
            : repMinimum(_min), repMaximum(_max) {}

        typedef double doubleype;

        double minimum(size_t i) { return repMinimum[i]; }
        double maximum(size_t i) { return repMaximum[i]; }

        unsigned chromSize(void) const { return repMinimum.size(); }

    private :
        std::vector<double> repMinimum;
        std::vector<double> repMaximum;
};

#endif