// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "FlowShop.h"

// (c) OPAC Team, LIFL, April 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/

#ifndef _FlowShop_h
#define _FlowShop_h

#include <EO.h>
// Fitness for multi-objective flow-shop
#include "FlowShopFitness.h"


/** 
 *  Structure of the genotype for the flow-shop scheduling problem
 */
class FlowShop:public EO < FlowShopFitness >
{

public:

  /** 
   * default constructor
   */
  FlowShop ()
  {
  }

  /** 
   * destructor
   */
  virtual ~ FlowShop ()
  {
  }

  /** 
   * class name
   */
  virtual string className () const
  {
    return "FlowShop";
  }

  /** 
   * set scheduling vector
   * @param vector<unsigned> & _scheduling  the new scheduling to set 
   */
  void setScheduling (vector < unsigned >&_scheduling)
  {
    scheduling = _scheduling;
  }

  /** 
   * get scheduling vector
   */
  const vector < unsigned >&getScheduling () const
  {
    return scheduling;
  }

  /**
   * printing...
   */
  void printOn (ostream & _os) const
  {
    // fitness
    EO < FlowShopFitness >::printOn (_os);
    _os << "\t";
    // size
    _os << scheduling.size () << "\t";
    // scheduling
    for (unsigned i = 0; i < scheduling.size (); i++)
      _os << scheduling[i] << ' ';
  }

  /**
   * reading...
   */
  void readFrom (istream & _is)
  {
    // fitness
    EO < FlowShopFitness >::readFrom (_is);
    // size
    unsigned size;
    _is >> size;
    // scheduling
    scheduling.resize (size);
    bool tmp;
    for (unsigned i = 0; i < size; i++)
      {
	_is >> tmp;
	scheduling[i] = tmp;
      }
  }


  bool operator== (const FlowShop & _other) const
  {
    return scheduling == _other.getScheduling ();
  }
  bool operator!= (const FlowShop & _other) const
  {
    return scheduling != _other.getScheduling ();
  }
  bool operator< (const FlowShop & _other) const
  {
    return scheduling < _other.getScheduling ();
  }
  bool operator> (const FlowShop & _other) const
  {
    return scheduling > _other.getScheduling ();
  }
  bool operator<= (const FlowShop & _other) const
  {
    return scheduling <= _other.getScheduling ();
  }
  bool operator>= (const FlowShop & _other) const
  {
    return scheduling >= _other.getScheduling ();
  }


private:

  /** scheduling (order of operations) */
    std::vector < unsigned >scheduling;

};

#endif
