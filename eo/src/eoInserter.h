/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoInserter.h
    Abstract population insertion operator, which is used by the eoGeneralOps
    to insert the results in the (intermediate) population. It also contains
    a direct descended eoPopInserter that defines a convenient inbetween class
    for working with eoPop<EOT>. The user will most likely derive from eoPopInserter
    rather than eoInserter.

 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000
 
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

#ifndef eoInserter_h
#define eoInserter_h

#include <eoFunctor.h>
#include <eoPop.h>

 /**
 * eoInserter: Interface class that enables an operator to insert
    new individuals into the (intermediate) population for example. 
*/
template <class EOT>
class eoInserter  : public eoUnaryFunctor<eoInserter<EOT>&, const EOT&>
{
    public :
        virtual ~eoInserter() {}

        struct eoInserterException{};
};

/**
 * eoPopInserter: In-between class that defines an initialization
 * of the eoIndividualInserter. 
*/
template <class EOT>
class eoPopInserter : public eoInserter<EOT> 
{
    public :
    
    eoPopInserter(void) : eoInserter<EOT>(), thePop(0) {}

    /// Binds the population to this class. This is an initialization routine used by breeders
    eoInserter<EOT>& bind(eoPop<EOT>& _pop)
    {
        thePop = &_pop;
        return *this;
    }

    protected :
        
    eoPop<EOT>& pop(void) const { valid(); return *thePop; }

    private :
    
    void valid(void) const
    {
        if (thePop == 0)
            throw eoInserterException();
    }

    // Need a pointer as the inserter should be able to bind to different populations.
    // This is caused by the 'one template parameter only' convention in EO.
        
    eoPop<EOT>* thePop; 
    
    // If eoGOpBreeder could be templatized over the inserter and the selector,
    // the pop could be a ref as this class could be created every time it is applied
    // and subsequently would get the population through the constructor

};



#endif

