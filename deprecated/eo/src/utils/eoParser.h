/* (c) Marc Schoenauer, Maarten Keijzer, GeNeura Team, Thales group

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this library; if not, write to the Free Software Foundation, Inc., 59
Temple Place, Suite 330, Boston, MA 02111-1307 USA

Contact: http://eodev.sourceforge.net
Authors:
    todos@geneura.ugr.es, http://geneura.ugr.es
    Marc.Schoenauer@polytechnique.fr
    mkeijzer@dhi.dk
    Johann Dréo <johann.dreo@thalesgroup.com>
*/


#ifndef EO_PARSER_H
#define EO_PARSER_H

#include <map>
#include <sstream>
#include <string>

#include "eoParam.h"
#include "eoObject.h"
#include "eoPersistent.h"
#include "eoExceptions.h"

/** Parameter saving and loading

eoParameterLoader is an abstract class that can be used as a base for your own
parameter loading and saving. The command line parser eoParser is derived from
this class.

@ingroup Parameters
*/
class eoParameterLoader
{
public :

    /** Need a virtual destructor */
    virtual ~eoParameterLoader();

    /** Register a parameter and set its value if it is known

    @param param      the parameter to process
    @param section    the section where this parameter belongs
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
     * @param _shortHand          Short name of the argument (Optional)
     * @param _section            Name of the section where the parameter belongs
     * @param _required           If it is a necessary parameter or not
     */
    template <class ValueType>
    eoValueParam<ValueType>& createParam(ValueType _defaultValue,
                                         std::string _longName,
                                         std::string _description,
                                         char _shortHand = 0,
                                         std::string _section = "",
                                         bool _required = false)
        {
            eoValueParam<ValueType>* p = new eoValueParam<ValueType>(_defaultValue,
                                                                     _longName,
                                                                     _description,
                                                                     _shortHand,
                                                                     _required);
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

    Parameters can be read from argv, strings or streams, and must be specified
    using the following convention: --name=value or -n=value

    You should not use space as a separator between the parameter and its value.

    @ingroup Parameters
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
  eoParser ( unsigned _argc, char **_argv , std::string _programDescription = "",
           std::string _lFileParamName = "param-file", char _shortHand = 'p');

  /**
    Processes the parameter and puts it in the appropriate section for readability
  */
  void processParam(eoParam& param, std::string section = "");

  /** Read from a stream
   * @param is the input stream
   */
  void readFrom(std::istream& is);

  /** Pint on a stream
   * @param os the output stream
   */
  void printOn(std::ostream& os) const;

  /// className for readibility
  std::string className(void) const { return "Parser"; }

  /// true if the user made an error or asked for help
  bool userNeedsHelp(void);
  /**
   * Prints an automatic help in the specified output using the information
   * provided by parameters
   */
  void printHelp(std::ostream& os);

  std::string ProgramName() { return programName; }

    /** Has param been entered by user?

    Checks if _param has been actually entered by the user
    */
    virtual bool isItThere(eoParam& _param) const
        { return getValue(_param).first; }


    std::string get( const std::string & name) const;


    /**
     * get a handle on a param from its longName
     *
     * if not found, returns 0 (null pointer :-)
     *
     * Not very clean (requires hard-coding of the long name twice!)
     * but very useful in many occasions...
     */
    eoParam * getParamWithLongName(const std::string& _name) const;


    /**
     * Get a handle on a param from its long name
     * If not found, raise an eoMissingParamException
     */
    eoParam * getParam(const std::string& _name) const;


    /**
     * Get the value of a param from its long name
     * If not found, raise an eoMissingParamException
     *
     * Remember to specify the expected return type with a templated call:
     * unsigned int popSize = eoparser.value<unsigned int>("popSize");
     *
     * If the template type is not the good one, an eoWrongParamTypeException is raised.
     */
    template<class ValueType>
    ValueType valueOf(const std::string& _name) const
    {
        eoParam* param = getParam(_name);

        // Note: eoParam is the polymorphic base class of eoValueParam, thus we can do a dynamix cast
        eoValueParam<ValueType>* vparam = dynamic_cast< eoValueParam<ValueType>* >(param);

        if( vparam == NULL ) {
            // if the dynamic cast has failed, chances are that ValueType 
            // is not the same than the one used at declaration.
            throw eoWrongParamTypeException( _name );
        } else {
            return vparam->value();
        }
    }



    /** Get or create parameter

    It seems finally that the easiest use of the above method is
    through the following, whose interface is similar to that of the
    widely-used createParam.
    */
    template <class ValueType>
    eoValueParam<ValueType>& getORcreateParam(ValueType _defaultValue,
                                              std::string _longName,
                                              std::string _description,
                                              char _shortHand = 0,
                                              std::string _section = "",
                                              bool _required = false)
        {
            eoParam* ptParam = getParamWithLongName(_longName);
            if (ptParam) {
                // found
                eoValueParam<ValueType>* ptTypedParam(
                    dynamic_cast<eoValueParam<ValueType>*>(ptParam));
                return *ptTypedParam;
            } else {
                // not found -> create it
                return createParam(_defaultValue, _longName, _description,
                                   _shortHand, _section, _required);
            }
        }



    /** Set parameter value or create parameter

    This makes sure that the specified parameter has the given value.
    If the parameter does not exist yet, it is created.

    This requires that operator<< is defined for ValueType.


    @param _defaultValue Default value.
    @param _longName     Long name of the argument.
    @param _description  Description of the parameter.
    @param _shortHand    Short name of the argument (Optional)
    @param _section      Name of the section where the parameter belongs.
    @param _required     Is the parameter mandatory?
    @return Corresponding parameter.
    */
    template <class ValueType>
    eoValueParam<ValueType>& setORcreateParam(ValueType _defaultValue,
                                              std::string _longName,
                                              std::string _description,
                                              char _shortHand = 0,
                                              std::string _section = "",
                                              bool _required = false)
        {
            eoValueParam<ValueType>& param = createParam(_defaultValue, _longName, _description,
                                                         _shortHand, _section, _required);
            std::ostringstream os;
            os << _defaultValue;
            if(isItThere(param)) {
                param.setValue(os.str());
            } else {
                longNameMap[_longName] = os.str();
                shortNameMap[_shortHand] = os.str();
            }
            return param;
        }



    /** accessors to the stopOnUnknownParam value */
    void setStopOnUnknownParam(bool _b) {stopOnUnknownParam.value()=_b;}
    bool getStopOnUnknownParam() {return stopOnUnknownParam.value();}

    /** Prefix handling */
    void setPrefix(const std:: string & _prefix) {prefix = _prefix;}

  void resetPrefix() {prefix = "";}

  std::string getPrefix() {return prefix;}

private:

  void doRegisterParam(eoParam& param);

  std::pair<bool, std::string> getValue(eoParam& _param) const;

  void updateParameters();

  typedef std::multimap<std::string, eoParam*> MultiMapType;

  // used to store all parameters that are processed
  MultiMapType params;

  std::string programName;
  std::string programDescription;

  typedef std::map<char, std::string> ShortNameMapType;
  ShortNameMapType shortNameMap;

  typedef std::map<std::string, std::string> LongNameMapType;
  LongNameMapType longNameMap;

  // flag that marks if the user need to know that there was a problem
  // used to display the message about "-h" only once
  bool needHelpMessage;

  eoValueParam<bool>   needHelp;
  eoValueParam<bool>   stopOnUnknownParam;

  mutable std::vector<std::string> messages;

  std::string prefix;   // used for all created params - in processParam

};



#endif //  EO_PARSER_H



// Local Variables:
// coding: iso-8859-1
// mode:C++
// c-file-style: "Stroustrup"
// comment-column: 35
// fill-column: 80
// End:
