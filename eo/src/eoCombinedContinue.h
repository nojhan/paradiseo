// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCombinedContinue.h
// (c) Maarten Keijzer, GeNeura Team, 1999, 2000
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
Authors :
    todos@geneura.ugr.es
    Marc Schoenauer
    Ramón Casero Cañas
    Johann Dréo
 */
//-----------------------------------------------------------------------------

#ifndef _eoCombinedContinue_h
#define _eoCombinedContinue_h

#include <eoContinue.h>

/**
    Combined continuators - logical AND:
  Continues until one of the embedded continuators says halt!

20/11/00 MS: Changed the 2-continuator construct to a std::vector<eoContinue<EOT> >
             to be consistent with other Combined constructs
             and allow to easily handle more than 2 continuators

02/2003 Ramón Casero Cañas - added the removeLast() method

@ingroup Combination
*/
template< class EOT>
class eoCombinedContinue: public eoContinue<EOT>, public std::vector<eoContinue<EOT>* > {
public:

  /// Define Fitness
  typedef typename EOT::Fitness FitnessType;

  /// Ctor, make sure that at least on continuator is present
  eoCombinedContinue( eoContinue<EOT>& _cont)
    : eoContinue<EOT>(), std::vector<eoContinue<EOT>* >(1, &_cont) 
  {
  }

  /* FIXME remove in next release
  /// Ctor - for historical reasons ... should disspear some day
  eoCombinedContinue( eoContinue<EOT>& _cont1, eoContinue<EOT>& _cont2)
    : eoContinue<EOT>(), std::vector<eoContinue<EOT>* >()
  {
#ifndef DEPRECATED_MESSAGES
#pragma message "The double continuators constructor of eocombinedContinue is deprecated and will be removed in the next release."
#endif // !DEPRECATED_MESSAGES

    this->push_back(&_cont1);
    this->push_back(&_cont2);
  }
  */

  void add(eoContinue<EOT> & _cont)
  {
    this->push_back(&_cont);
  }

  /* FIXME remove in next release
  void removeLast(void)
  {
#ifndef DEPRECATED_MESSAGES
#pragma message "The removeLast method of eocombinedContinue is deprecated and will be removed in the next release, use pop_back instead."
#endif // !DEPRECATED_MESSAGES

    this->pop_back();
  }
  */


  /** Returns false when one of the embedded continuators say so (logical and)
   */
  virtual bool operator() ( const eoPop<EOT>& _pop )
  {
    for (unsigned i = 0; i < this->size(); ++i)
      if ( ! (*this->at(i))(_pop) ) return false;
    return true;
  }

  virtual std::string className(void) const { return "eoCombinedContinue"; }

//private:
//  std::vector<eoContinue<EOT>*>    continuators;
};

#endif
