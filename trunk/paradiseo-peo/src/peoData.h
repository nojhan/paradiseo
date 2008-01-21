/*
* <peoData.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Clive Canape, Thomas Legrand
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* peoData to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef _PEODATA_H
#define _PEODATA_H

#include "core/eoVector_mesg.h"
#include "core/messaging.h"

/**************************************************************************************/
/**************************  DEFINE A DATA   ******************************************/
/**************************************************************************************/

class peoData {
public:

  virtual void pack () {}
  virtual void unpack () {}
  
};


// Specific implementation : migration of a population

template<class EOT>
class peoPop: public eoPop<EOT>, public peoData
{
public:

  virtual void pack () 
  {
  	 ::pack ((unsigned) this->size ());
 	 for (unsigned i = 0; i < this->size (); i ++)
    	::pack ((*this)[i]);
  }

  virtual void unpack () 
  {
	  unsigned n;
	  ::unpack (n);
	  this->resize (n);
	  for (unsigned i = 0; i < n; i ++)
	  ::unpack ((*this)[i]);
  }

};


/**************************************************************************************/
/**************************  DEFINE A CONTINUATOR   ***********************************/
/**************************************************************************************/

class continuator
{
public: 	
	virtual bool check()=0;
};


// Specific implementation : migration of a population

template < class EOT> class eoContinuator : public continuator{
public:
	
	eoContinuator(eoContinue<EOT> & _cont, const eoPop<EOT> & _pop): cont (_cont), pop(_pop){}
	
	virtual bool check(){
		return cont(pop);	
	}

protected:
	eoContinue<EOT> & cont ;
	const eoPop<EOT> & pop;
};


/**************************************************************************************/
/**************************  DEFINE A SELECTOR   **************************************/
/**************************************************************************************/

template < class TYPE>  class selector
{
public: 	
	virtual void operator()(TYPE &)=0;
};


// Specific implementation : migration of a population

template < class EOT, class TYPE> class eoSelector : public selector< TYPE >{
public:
	
	eoSelector(eoSelectOne<EOT> & _select, unsigned _nb_select, const TYPE & _source): selector (_select), nb_select(_nb_select), source(_source){}
	
	virtual void operator()(TYPE & _dest)
  	{
    	size_t target = static_cast<size_t>(nb_select);
    	_dest.resize(target);
       	for (size_t i = 0; i < _dest.size(); ++i)
      		_dest[i] = selector(source);
  	}

protected:
	eoSelectOne<EOT> & selector ;
	unsigned nb_select;
	const TYPE & source;
};


/**************************************************************************************/
/**************************  DEFINE A REPLACEMENT   ***********************************/
/**************************************************************************************/

template < class TYPE>  class replacement
{
public: 	
	virtual void operator()(TYPE &)=0;
};


// Specific implementation : migration of a population

template < class EOT, class TYPE> class eoReplace : public replacement< TYPE >{
public:
	eoReplace(eoReplacement<EOT> & _replace, TYPE & _destination): replace(_replace), destination(_destination){} 
	
	virtual void operator()(TYPE & _source)
  	{
    	replace(destination, _source);
  	}

protected:
	eoReplacement<EOT> & replace;
	TYPE & destination;
};


/**************************************************************************************/
/************************  Continuator for synchrone migartion ************************/
/**************************************************************************************/

class eoSyncContinue: public continuator
{

public:
    
  eoSyncContinue (unsigned __period, unsigned __init_counter = 0): period (__period),counter (__init_counter) {}
   
  virtual bool check()
  {
  	return ((++ counter) % period) != 0 ;
  }
  
     
private:

  unsigned period;
  
  unsigned counter;

};


#endif

