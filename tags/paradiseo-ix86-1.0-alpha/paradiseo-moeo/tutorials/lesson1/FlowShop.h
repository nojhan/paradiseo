// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShop.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOP_H_
#define FLOWSHOP_H_

#include <MOEO.h>
#include <moeoObjectiveVector.h>
#include <moeoObjectiveVectorTraits.h>


/**
 * definition of the objective vector for multi-objective flow-shop problems
 */
typedef moeoObjectiveVectorDouble<moeoObjectiveVectorTraits> FlowShopObjectiveVector;


/** 
 *  Structure of the genotype for the flow-shop scheduling problem
 */
class FlowShop: public MOEO<FlowShopObjectiveVector, double, double> {

public:

  /** 
   * default constructor
   */
  FlowShop() {}

  /** 
   * destructor
   */
  virtual ~FlowShop() {}
  
  /** 
   * class name
   */
  virtual string className() const { 
    return "FlowShop";
  }

  /** 
   * set scheduling vector
   * @param vector<unsigned> & _scheduling  the new scheduling to set 
   */
  void setScheduling(vector<unsigned> & _scheduling) {
    scheduling = _scheduling;
  }
  
  /** 
   * get scheduling vector
   */
  const vector<unsigned> & getScheduling() const {
    return scheduling;
  }
  
  /**
   * printing...
   */
  void printOn(ostream& _os) const {
    // fitness
    MOEO<FlowShopObjectiveVector, double, double>::printOn(_os);    
    // size
    _os << scheduling.size() << "\t" ;
    // scheduling
    for (unsigned i=0; i<scheduling.size(); i++)
      _os << scheduling[i] << ' ' ;
  }
  
  /**
   * reading...
   */
  void readFrom(istream& _is) {
    // fitness
    MOEO<FlowShopObjectiveVector, double, double>::readFrom(_is);
    // size
    unsigned size;
    _is >> size;
    // scheduling
    scheduling.resize(size);
    bool tmp;
    for (unsigned i=0; i<size; i++) {
      _is >> tmp;
      scheduling[i] = tmp;
    }
  }
  
  
  bool operator==(const FlowShop& _other) const { return scheduling == _other.getScheduling(); }
  bool operator!=(const FlowShop& _other) const { return scheduling != _other.getScheduling(); }
  bool operator< (const FlowShop& _other) const { return scheduling <  _other.getScheduling(); }
  bool operator> (const FlowShop& _other) const { return scheduling >  _other.getScheduling(); }
  bool operator<=(const FlowShop& _other) const { return scheduling <= _other.getScheduling(); }
  bool operator>=(const FlowShop& _other) const { return scheduling >= _other.getScheduling(); }


private:

  /** scheduling (order of operations) */
  std::vector<unsigned> scheduling;

};


#endif /*FLOWSHOP_H_*/
