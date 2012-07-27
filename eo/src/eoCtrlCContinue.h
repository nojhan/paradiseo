// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoCtrlCContinue.h
// (c) EEAAX 1996 - GeNeura Team, 1998 - Maarten Keijzer 2000
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
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
*/
//-----------------------------------------------------------------------------
// the same thing can probably be done in MS environement, but I hoave no way
// to test it so at the moment it is commented out when in MSVC


#ifndef eoCtrlCContinue_h
#define eoCtrlCContinue_h

#include <csignal>
#include <eoContinue.h>

/**
 * @addtogroup Continuators
 * @{
 */

extern bool ask_for_stop, existCtrlCContinue;

extern void signal_handler( int sig );


/**
    Ctrl C handling: this eoContinue tells whether the user pressed Ctrl C
*/
template< class EOT>
class eoCtrlCContinue: public eoContinue<EOT>
{
public:

  /// Ctor : installs the signal handler
  eoCtrlCContinue()
  {
    // First checks that no other eoCtrlCContinue does exist
    if (existCtrlCContinue)
      throw std::runtime_error("A signal handler for Ctrl C is already defined!\n");

    #ifndef _WINDOWS
      #ifdef SIGQUIT
        signal( SIGINT, signal_handler );
        signal( SIGQUIT, signal_handler );
        existCtrlCContinue = true;
      #endif
    #endif

  }

  /** Returns false when Ctrl C has been typed in
         * reached */
  virtual bool operator() ( const eoPop<EOT>& _vEO )
  {
      (void)_vEO;
    if (ask_for_stop)
      return false;
    return true;
  }

  virtual std::string className(void) const { return "eoCtrlCContinue"; }
};


#endif
/** @} */
