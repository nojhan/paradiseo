// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsBase.h
// (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000
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

#ifndef _eoFixedLength_h
#define _eoFixedLength_h

/**
\ingroup EvolutionStrategies

  Base class for fixed length chromosomes, just derives from EO and vector and
  redirects the smaller than operator to EO (fitness based comparison)
*/

template <class FitT, class GeneType>
class eoFixedLength : public EO<FitT>, public std::vector<GeneType>
{
    public :
    
    typedef double Type;

    /// to avoid conflicts between EO::operator< and vector<double>::operator<
    bool operator<(const eoFixedLength<FitT, GeneType>& _eo) const
    {
        return EO<FitT>::operator<(_eo);
    }

};

/// to avoid conflicts between EO::operator< and vector<double>::operator<
template <class FitT, class GeneType>
bool operator<(const eoFixedLength<FitT, GeneType>& _eo1, const eoFixedLength<FitT, GeneType>& _eo2)
{
    return _eo1.operator<(_eo2);
}

#endif
