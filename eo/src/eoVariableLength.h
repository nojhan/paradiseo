// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVariableLength.h
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

#ifndef _eoVariableLength_h
#define _eoVariableLength_h

#include <vector>

/**
  Base class for variable length chromosomes. Derives from EO and vector, 
  redirects the smaller than operator to EO (fitness based comparison),
  and implements the virtual functions printOn() and readFrom() 
*/

template <class FitT, class GeneType>
class eoVariableLength : public EO<FitT>, public std::vector<GeneType>
{
    public :
    
    typedef GeneType AtomType;
    typedef std::vector<GeneType> ContainerType;

  // default ctor
    eoVariableLength(unsigned size = 0, GeneType value = GeneType()) : EO<FitT>(), std::vector<GeneType>(size, value)
    {}

  // we can't have a Ctor from a vector, it would create ambiguity
  //  with the copy Ctor
  void value(std::vector<GeneType> _v)
  {
    resize(_v.size());
    copy(_v.begin(), _v.end(), begin());
    invalidate();
  }

    /// printing...
    void printOn(ostream& os) const
    {
        EO<FitT>::printOn(os);
        os << ' ';

        os << size() << ' ';

        std::copy(begin(), end(), ostream_iterator<AtomType>(os, " "));
    }

    /// reading...
    void readFrom(istream& is)
    {
        EO<FitT>::readFrom(is);

        unsigned sz;
        is >> sz;

        resize(sz);
        unsigned i;

        for (i = 0; i < sz; ++i)
        {
            AtomType atom;
            is >> atom;
            operator[](i) = atom;
        }
    }
    
    /// to avoid conflicts between EO::operator< and vector<double>::operator<
    bool operator<(const eoVariableLength<FitT, GeneType>& _eo) const
    {
        return EO<FitT>::operator<(_eo);
    }

};

/// to avoid conflicts between EO::operator< and vector<GeneType>::operator<
template <class FitT, class GeneType>
bool operator<(const eoVariableLength<FitT, GeneType>& _eo1, const eoVariableLength<FitT, GeneType>& _eo2)
{
    return _eo1.operator<(_eo2);
}

#endif
