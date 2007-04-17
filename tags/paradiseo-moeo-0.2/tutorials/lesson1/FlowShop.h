// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShop.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

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
