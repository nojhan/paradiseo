// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParser.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
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
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoParser_h
#define eoParser_h

#include <map>
#include <string>
#include <map>

#include "eoParam.h"
#include "eoObject.h"
#include "eoPersistent.h"

/**
    eoParameterLoader is an abstract class that can be used as a base for your own 
    parameter loading and saving. The command line parser eoParser is derived from
    this class.
*/
class eoParameterLoader
{
public :
    
    /** Need a virtual destructor */
    virtual ~eoParameterLoader() {}

    /**
      *  processParam is used to register a parameter and set its value if it is known
      *     
      *   @param param      the parameter to process
      *   @param section    the section where this parameter belongs
    */
    virtual void processParam(eoParam& param, std::string section = "") = 0;
};

/**
    eoParser: command line parser and configuration file reader
    This class is persistent, so it can be stored and reloaded to restore
    parameter settings.
*/
class eoParser : public eoParameterLoader, public eoObject, public eoPersistent
{

public:

  /**
   * Constructor
   * a complete constructor that reads the command line an optionally reads
   * a configuration file.
   *
   * myEo --param-file=param.rc     will then load using the parameter file param.rc
   *
   * @param _argc, _ argv           command line arguments
   * @param  _programDescription    Description of the work the program does
   * @param _lFileParamName         Name of the parameter specifying the configuration file (--param-file)
   * @param _shortHand              Single charachter shorthand for specifying the configuration file
   */
  eoParser ( int _argc, char **_argv , string _programDescription = "", 
	   string _lFileParamName = "param-file", char _shortHand = 'p');  
  
  /**
    Processes the parameter and puts it in the appropriate section for readability
  */
  void processParam(eoParam& param, std::string section = "");

  void readFrom(istream& is);

  void printOn(ostream& os) const;
  
  /// className for readibility 
  std::string className(void) const { return "Parser"; }

  /// true if the user made an error or asked for help
  bool userNeedsHelp(void) const { return needHelp.value(); }

  /**
   * Prints an automatic help in the specified output using the information
   * provided by parameters
   */
  void printHelp(ostream& os);

  string ProgramName() { return programName; }
 
private:
  
  void doRegisterParam(eoParam& param) const;
  
  bool isItThere(eoParam& _param) const { return getValue(_param).first; }

  std::pair<bool, string> getValue(eoParam& _param) const;

  void updateParameters() const;
  
  typedef std::multimap<std::string, eoParam*> MultiMapType;
  
  MultiMapType params;
  
  string programName; 
  string programDescription;

  map<char, string>   shortNameMap;
  map<string, string> longNameMap;

  eoValueParam<string> parameterFile;
  eoValueParam<bool>   needHelp;
};

#endif