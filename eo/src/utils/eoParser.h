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
/**
CVS Info: $Date: 2002-09-18 15:36:41 $ $Version$ $Author: evomarc $
*/
#ifndef eoParser_h
#define eoParser_h

#include <map>
#include <string>

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
    virtual ~eoParameterLoader();

    /**
      *  processParam is used to register a parameter and set its value if it is known
      *     
      *   @param param      the parameter to process
      *   @param section    the section where this parameter belongs
    */
    virtual void processParam(eoParam& param, std::string section = "") = 0;

  /** 
   * checks if _param has been actually entered
   */
  virtual bool isItThere(eoParam& _param) const = 0;

  /**
   * Construct a Param and sets its value. The loader will own the memory thus created
   *
   * @param _defaultValue       The default value
   * @param _longName           Long name of the argument
   * @param _description        Description of the parameter. What is useful for.
   * @param _shortName          Short name of the argument (Optional)
   * @param _section            Name of the section where the parameter belongs
   * @param _required           If it is a necessary parameter or not
   */
    template <class ValueType>
    eoValueParam<ValueType>& createParam
               (ValueType _defaultValue, 
                std::string _longName, 
                std::string _description,
                char _shortHand = 0,
                std::string _section = "",
                bool _required = false)
    {
        eoValueParam<ValueType>* p = new eoValueParam<ValueType>(_defaultValue, _longName, _description, _shortHand, _required);

        ownedParams.push_back(p);
    
        processParam(*p, _section);

        return *p;
    }

private :

    std::vector<eoParam*> ownedParams;

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
   * @param _argc                   command line arguments count
   * @param _argv                   command line parameters
   * @param  _programDescription    Description of the work the program does
   * @param _lFileParamName         Name of the parameter specifying the configuration file (--param-file)
   * @param _shortHand              Single charachter shorthand for specifying the configuration file
   */
  eoParser ( unsigned _argc, char **_argv , string _programDescription = "", 
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
  bool userNeedsHelp(void); 
  /**
   * Prints an automatic help in the specified output using the information
   * provided by parameters
   */
  void printHelp(ostream& os);

  string ProgramName() { return programName; }
 
  /** 
   * checks if _param has been actually entered by the user
   */
  virtual bool isItThere(eoParam& _param) const 
  { return getValue(_param).first; }

/** 
 * get a handle on a param from its longName
 * 
 * if not found, returns 0 (null pointer :-)
 *
 * Not very clean (requires hard-coding of the long name twice!)
 * but very useful in many occasions...
 */
  eoParam* getParamWithLongName(std::string _name);

  /** it seems finally that the easiest use of the above method is
      through the following, whose interface is similar to that of the
      widely-used createParam
      For some (probably very stupid) reason, I failed to put it in
      the .cpp. Any hint???
  */
    template <class ValueType>
    eoValueParam<ValueType>& getORcreateParam
               (ValueType _defaultValue, 
                std::string _longName, 
                std::string _description,
                char _shortHand = 0,
                std::string _section = "",
                bool _required = false)
{
  eoParam* ptParam = getParamWithLongName(_longName);
  if (ptParam) {			// found
    eoValueParam<ValueType>* ptTypedParam = 
      dynamic_cast<eoValueParam<ValueType>*>(ptParam);
    return *ptTypedParam;
  }
  // not found -> create it
  return createParam (_defaultValue, _longName, _description, 
		      _shortHand, _section, _required);
}

//   /** accessors to the stopOnUnknownParam value */
  void setStopOnUnknownParam(bool _b) {stopOnUnknownParam.value()=_b;}
  bool getStopOnUnknownParam() {return stopOnUnknownParam.value();}


private:
  
  void doRegisterParam(eoParam& param) const;
  
  std::pair<bool, string> getValue(eoParam& _param) const;

  void updateParameters() const;
  
  typedef std::multimap<std::string, eoParam*> MultiMapType;

  // used to store all parameters that are processed
  MultiMapType params;
  
  string programName; 
  string programDescription;

  typedef map<char, string> ShortNameMapType;
  ShortNameMapType shortNameMap;
  
  typedef map<string, string> LongNameMapType;
  LongNameMapType longNameMap;

  eoValueParam<bool>   needHelp;
  eoValueParam<bool>   stopOnUnknownParam;

  mutable std::vector<std::string> messages;
};


#endif
