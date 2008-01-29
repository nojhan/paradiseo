/*
* <peoData.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2008
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

//! @class peoData
//! @brief Abstract class for a data exchanged by migration
//! @version 1.0
//! @date january 2008
class peoData
  {
  public:
  
	//! @brief Function realizing packages
    virtual void pack ()
    {}
    
	//! @brief Function reconstituting packages
    virtual void unpack ()
    {}

  };

//! @class peoPop
//! @brief Specific class for a migration of a population
//! @see peoData eoPop
//! @version 1.0
//! @date january 2008
template<class EOT>
class peoPop: public eoPop<EOT>, public peoData
  {
  public:

	//! @brief Function realizing packages
    virtual void pack ()
    {
      ::pack ((unsigned) this->size ());
      for (unsigned i = 0; i < this->size (); i ++)
        ::pack ((*this)[i]);
    }

	//! @brief Function reconstituting packages
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

//! @class continuator
//! @brief Abstract class for a continuator within the exchange of data by migration
//! @version 1.0
//! @date january 2008
class continuator
  {
  public:
  
	 //! @brief Virtual function of check
	 //! @return true if the algorithm must continue
    virtual bool check()=0;
  };


//! @class eoContinuator
//! @brief Specific class for a continuator within the exchange of migration of a population
//! @see continuator
//! @version 1.0
//! @date january 2008
template < class EOT> class eoContinuator : public continuator
  {
  public:

	//! @brief Constructor
	//! @param eoContinue<EOT> & 
	//! @param eoPop<EOT> & 
    eoContinuator(eoContinue<EOT> & _cont, const eoPop<EOT> & _pop): cont (_cont), pop(_pop)
    {}

	//! @brief Virtual function of check
	//! @return true if the algorithm must continue
    virtual bool check()
    {
      return cont(pop);
    }

  protected:
  	 //! @param eoContinue<EOT> &
  	 //! @param eoPop<EOT> &
    eoContinue<EOT> & cont ;
    const eoPop<EOT> & pop;
  };


/**************************************************************************************/
/**************************  DEFINE A SELECTOR   **************************************/
/**************************************************************************************/

//! @class selector
//! @brief Abstract class for a selector within the exchange of data by migration
//! @version 1.0
//! @date january 2008
template < class TYPE>  class selector
  {
  public:
  	
  	//! @brief Virtual operator on the template type 
  	//! @param TYPE &
    virtual void operator()(TYPE &)=0;
  };


//! @class eoSelector
//! @brief Specific class for a selector within the exchange of migration of a population
//! @see selector
//! @version 1.0
//! @date january 2008
template < class EOT, class TYPE> class eoSelector : public selector< TYPE >
  {
  public:

	//! @brief Constructor
	//! @param eoSelectOne<EOT> &
	//! @param unsigned _nb_select
	//! @param TYPE & _source (with TYPE which is the template type)
    eoSelector(eoSelectOne<EOT> & _select, unsigned _nb_select, const TYPE & _source): selector (_select), nb_select(_nb_select), source(_source)
    {}
    
	//! @brief Virtual operator on the template type
	//! @param TYPE & _dest
	virtual void operator()(TYPE & _dest)
    {
      size_t target = static_cast<size_t>(nb_select);
      _dest.resize(target);
      for (size_t i = 0; i < _dest.size(); ++i)
        _dest[i] = selector(source);
    }

  protected:
  	//! @param eoSelectOne<EOT> &
  	//! @param unsigned nb_select
  	//! @param TYPE & source
    eoSelectOne<EOT> & selector ;
    unsigned nb_select;
    const TYPE & source;
  };


/**************************************************************************************/
/**************************  DEFINE A REPLACEMENT   ***********************************/
/**************************************************************************************/

//! @class replacement
//! @brief Abstract class for a replacement within the exchange of data by migration
//! @version 1.0
//! @date january 2008
template < class TYPE>  class replacement
  {
  public:
  	//! @brief Virtual operator on the template type 
  	//! @param TYPE &
    virtual void operator()(TYPE &)=0;
  };


//! @class eoReplace
//! @brief Specific class for a replacement within the exchange of migration of a population
//! @see replacement
//! @version 1.0
//! @date january 2008
template < class EOT, class TYPE> class eoReplace : public replacement< TYPE >
  {
  public:
  	//! @brief Constructor
	//! @param eoReplacement<EOT> &
	//! @param TYPE & _destination (with TYPE which is the template type)
    eoReplace(eoReplacement<EOT> & _replace, TYPE & _destination): replace(_replace), destination(_destination)
    {}

	//! @brief Virtual operator on the template type
	//! @param TYPE & _source
    virtual void operator()(TYPE & _source)
    {
      replace(destination, _source);
    }

  protected:
  	//! @param eoReplacement<EOT> &
  	//! @param TYPE & destination
    eoReplacement<EOT> & replace;
    TYPE & destination;
  };


/**************************************************************************************/
/************************  Continuator for synchrone migartion ************************/
/**************************************************************************************/

//! @class eoSyncContinue
//! @brief Class for a continuator within the exchange of data by synchrone migration
//! @see continuator
//! @version 1.0
//! @date january 2008
class eoSyncContinue: public continuator
  {

  public:
	//! @brief Constructor
	//! @param unsigned __period
	//! @param unsigned __init_counter
    eoSyncContinue (unsigned __period, unsigned __init_counter = 0): period (__period),counter (__init_counter)
    {}

	//! @brief Virtual function of check
	//! @return true if the algorithm must continue
    virtual bool check()
    {
      return ((++ counter) % period) != 0 ;
    }


  private:
	//! @param unsigned period
	//! @param unsigned counter
    unsigned period;
    unsigned counter;
  };


#endif

