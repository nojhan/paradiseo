// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoIslandsEasyEA"

// (c) OPAC Team, LIFL, 2002

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
   
   Cont	act: cahon@lifl.fr
*/

#ifndef eoIslandsEasyEA_h
#define eoIslandsEasyEA_h

#include <vector>
#include <string>
#include <paradisEO/comm/eoListener.h>
#include <paradisEO/island/eoConnectivity.h>
#include <paradisEO/island/eoMigUpdater.h>
#include <eoContinue.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <utils/eoCheckPoint.h>
#include <utils/eoUpdater.h>

/**
   An island distributed easy evolutionary algorithm.
   It embeds an instance of easyEA. The behavior of this
   last one isn't modified. However, exchanges of individuals
   are performed with other EAs.
 */

template <class EOT> class eoIslandsEasyEA : public eoEasyEA <EOT> {
  
public :
  
  /**
     Constructor
  */
  
  eoIslandsEasyEA (std::string _id,
		   eoListener <EOT> & _listen,
		   eoConnectivity <EOT> & _conn,
		   eoEasyEA <EOT> & _ea,
		   eoContinue <EOT> & _cont,
		   eoSelect <EOT> & _select,
		   eoReplacement <EOT> & _replace
		   ) :
    id (_id),
    listen (_listen),
    conn (_conn),
    chkp (_ea.continuator),
    mig_upd (_listen,
	     _conn,
	     _cont,
	     _select,
	     _replace),
    eoEasyEA <EOT> (chkp,
		    _ea.eval,
		    _ea.breed,
		    _ea.replace
		    ) {
    chkp.add (mig_upd) ;
  }
  
  virtual void operator () (eoPop <EOT> & pop) {
    
    mig_upd (pop) ; // Sets pop. to send/receive EO
    listen.publish (id) ;
    eoEasyEA <EOT> :: operator () (pop) ;
    listen.publish ("_") ;
  }

private :

  // Internal components
  
  std::string id ; // String identifiant of this algorithm
  eoListener <EOT> & listen ; // The neighbouring of concurrent algos
  eoConnectivity <EOT> & conn ; // Communication topology
  eoCheckPoint <EOT> chkp ;
  eoMigUpdater <EOT> mig_upd ;
} ;

#endif
