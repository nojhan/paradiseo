// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSIGContinue.h
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
             Johann Dréo <nojhan@gmail.com>
             Caner Candan <caner@candan.fr>
*/
//-----------------------------------------------------------------------------
// the same thing can probably be done in MS environement, but I hoave no way
// to test it so at the moment it is commented out when in MSVC


#ifndef eoSIGContinue_h
#define eoSIGContinue_h

#include <signal.h>
#include <eoContinue.h>

/** @addtogroup Continuators
 * @{
 */

extern bool existSIGContinue;
extern bool call_func;

void	set_bool(int)
{
  call_func = true;
}

/**
  A continuator that stops if a given signal is received during the execution
*/
template< class EOT>
class eoSIGContinue: public eoContinue<EOT>
{
public:
  /// Ctor : installs the signal handler
  eoSIGContinue(int sig, sighandler_t fct)
    : _sig(sig), _fct(fct)
  {
    // First checks that no other eoSIGContinue does exist
    if (existSIGContinue)
      throw std::runtime_error("A signal handler is already defined!\n");

    #ifndef _WINDOWS
      #ifdef SIGQUIT
        ::signal( sig, set_bool );
        existSIGContinue = true;
      #endif
    #endif
  }

  /** Returns false when the signal has been typed in reached */
  virtual bool operator() ( const eoPop<EOT>& _vEO )
  {
    if (call_func)
      {
        _fct(_sig);
        call_func = false;
      }

    return true;
  }

  virtual std::string className(void) const { return "eoSIGContinue"; }
private:
  int		_sig;
  sighandler_t	_fct;
};

/** @} */

#endif

 // of MSVC comment-out
