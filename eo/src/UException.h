//-----------------------------------------------------------------------------
// UException.h
//-----------------------------------------------------------------------------

#ifndef _UEXCEPTION_H
#define _UEXCEPTION_H

#include <string>
#if defined (__BCPLUSPLUS__)
#include <stdexcept>
#else
#include <exception>
#endif

using namespace std;

//-----------------------------------------------------------------------------
// Class UException
//-----------------------------------------------------------------------------

//@{
/**
 * This class manages exceptions. It´s barely an extension of the standard exception, 
 * but it can be initialized with an STL string. Called UException (utils-exception)+
 * to avoid conflicts with other classes.
 */
class UException: public exception {
 public:
  ///
  UException( const string& _msg ): msg( _msg ) { };

  ///
  virtual const char* what() const { return msg.c_str(); };

 private:
  string msg;
};


//@}
#endif
