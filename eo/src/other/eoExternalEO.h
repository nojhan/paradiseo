/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoExternalEO.h
        * Definition of an object that allows an external struct
        * to be inserted in EO  
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

#include <EO.h>

/** 
 * Definition of an object that allows an external struct
 * to be inserted in EO  
*/
template <class Fit, class External>
class eoExternalEO : public EO, virtual public External
{
    public :

        typedef External Type;

        eoExternalEO(void) : EO(), Base() {}
        eoExternalEO(istream& is) : EO(), Base() { readFrom(is); }

       /**
       * Read object.\\
       * @param _is a istream.
       * @throw runtime_exception If a valid object can't be read.
       */
      virtual void readFrom(istream& _is) 
      { 
          EO::readFrom(is);
          throw runtime_excpetion("Reading not defined yet");
      }
  
      /**
       * Write object. Called printOn since it prints the object _on_ a stream.
       * @param _os A ostream.
       */
      virtual void printOn(ostream& _os) const 
      {
          EO::printOn(is);
          throw runtime_excpetion("Writing not defined yet");
      }

};

#endif
