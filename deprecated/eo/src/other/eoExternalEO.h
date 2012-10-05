/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoExternalEO.h
        Definition of an object that allows an external struct to be inserted in EO

 (c) Maarten Keijzer (mkeijzer@mad.scientist.com) and GeNeura Team, 1999, 2000

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
 */

#ifndef eoExternalEO_h
#define eoExternalEO_h

#include <EO.h> // EO

/**
 * Definition of an object that allows an external struct
 * to be inserted in EO. This struct or class can be of any
 * form, the only thing this class does is attach a fitness
 * value to it and makes it the appropriate type (derives it from EO).
 *
 * @ingroup Utilities
*/
template <class Fit, class External>
class eoExternalEO : public EO<Fit>, virtual public External
{
public :

    eoExternalEO()
        : External(), EO<Fit>()
        {}

    /** Init externalEo with the struct itself and set fitness to zero */
    eoExternalEO(const External& ext)
        : EO<Fit>(), External(ext)
        {}

    eoExternalEO(std::istream& is, const External& ext)
        : EO<Fit>(), External(ext)
        { readFrom(is); }

  /**
   * Read object, the external struct needs to have an operator>> defined
   */
  virtual void readFrom(std::istream& _is)
  {
      EO<Fit>::readFrom(_is);
    _is >> static_cast<External&>(*this);
  }

  /**
   * Write object. Called printOn since it prints the object _on_ a stream.
   * @param _os A std::ostream.
   */
  virtual void printOn(std::ostream& _os) const
  {
      EO<Fit>::printOn(_os);
      _os << static_cast<const External&>(*this);
  }

};
/** @example t-eoExternalEO.cpp
 */

/** To remove ambiguities between EO<F> and External, streaming operators are defined yet again
 * @ingroup Utilities
 */
template <class F, class External>
std::ostream& operator<<(std::ostream& os, const eoExternalEO<F, External>& eo)
{
    eo.printOn(os);
    return os;
}

/** To remove ambiguities between EO<F> and External, streaming operators are defined yet again
 * @ingroup Utilities
 */
template <class F, class External>
std::istream& operator>>(std::istream& is, eoExternalEO<F, External>& eo)
{
    eo.readFrom(is);
    return is;
}

#endif
