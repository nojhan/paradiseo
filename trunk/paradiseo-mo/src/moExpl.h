/*
 <moExpl.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
 (C) OPAC Team, LIFL, 2002-2008

 Salma Mesmoudi (salma.mesmoudi@inria.fr), Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */
#ifndef _moExpl_h
#define _moExpl_h

#include <eoFunctor.h>

//! Description of an explorer
/*!
  This class allows to use any mutation object as a neighborhood.
*/
template < class EOT >
class moExpl : public eoBF < const EOT &, EOT &, bool >
{
 public:
  
  //! Generic constructor
  /*!
    Generic constructor using a eoMonOp

    \param _explorer Algorithme or mutation.

  */
 moExpl(eoMonOp<EOT> & _explorer): index(0)
    {
      explorers.clear();
      addExplorer(_explorer);
    }
  
  //!  Procedure which launches the moExpl.
  /*!
    The exploration starts from an old solution and provides a new solution.
    
    \param _old_solution The current solution.
    \param _new_solution The new solution (result of the procedure).
  */

  bool operator ()(const EOT & _old_solution, EOT & _new_solution)
  {
    _new_solution=(EOT)_old_solution;
    return (*explorers[index])(_new_solution);
  }
  
  //! Add an algorithm or mutation to neighborhoods vector
  void addExplorer(eoMonOp<EOT> & _new_explorer)
  {
    explorers.push_back(&_new_explorer);
  }
  
  //! Procedure which modified the current explorer to use.
  /*!
    \param _index Index of the explorer to use.
   */
  void setCurrentExplorer(unsigned int _index)
  {
    if( _index >= explorers.size() )
      {
	std::cout << "[" << _index << "]" << std::endl;
	throw std::runtime_error("[moExpl.h]: bad index "+_index);
      }
    index=_index;
  }
  
  //! Function which returns the number of explorers already saved.
  /*!
    \return The number of explorers contained in the moExpl.
   */
  unsigned int getExplorerNumber()
  {
    return explorers.size();
  }
  
 private :

  unsigned int index;

  //!Neighborhoods vector
  std::vector< eoMonOp<EOT>* > explorers;
};

#endif
