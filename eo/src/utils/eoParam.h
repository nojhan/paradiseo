// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParam.h
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

#ifndef eoParam_h
#define eoParam_h

//-----------------------------------------------------------------------------
#include <string>
#include <strstream>

/**
    eoParam: Base class for monitoring and parsing parameters
*/
class eoParam 
{
public:
  
  /** Empty constructor - called from outside any parser
   */
  eoParam ()
    : repLongName(""), repDescription(""), repDefault(""), 
    repShortHand(0), repRequired(false){}

  /**
   * Construct a Param.
   * @param _longName      Long name of the argument
   * @param _default       The default value
   * @param _description   Description of the parameter. What is useful for.
   * @param _shortName     Short name of the argument (Optional)
   * @param _required      If it is a necessary parameter or not
   */
  eoParam (std::string _longName, std::string _default, 
      std::string _description, char _shortName = 0, bool _required = false)
    : repShortHand(_shortName), repLongName(_longName), 
    repDescription(_description ), repDefault(_default),
    repRequired( _required) {}
  
  /**
   * Virtual destructor is needed.
   */
  virtual ~eoParam () {};
  
  /**
  * Pure virtual function to get the value out.
  */
  virtual std::string getValue ( void ) const = 0;

  /**
  * Pure virtual function to set the value
  */
  virtual void setValue(std::string _value)   = 0 ; 
 
  /**
   * Returns the short name.
   */
  char shortName ( void ) const { return repShortHand; };
  
  /**
   * Returns the long name.
   */
  const std::string& longName ( void ) const { return repLongName; };
  
  /**
   * Returns the description of the argument
   */
  const std::string& description ( void ) const { return repDescription; };
  
  /**
   * Returns the default value of the argument
   */
  const std::string& defValue ( void ) const { return repDefault; };
  
  /**
   * Sets the default value of the argument, 
   */
  void defValue ( std::string str ) { repDefault = str; };
  
  /**
   * Returns the value of the param as a string
   */
  /**
   * Returns if required or not.
   */
  bool required ( void ) const { return repRequired; };
   
private:
    std::string repLongName;
    std::string repDefault;
    std::string repDescription;
  char   repShortHand;  
  bool repRequired;
};

/**
    eoValueParam<ValueType>: templatized derivation of eoParam. Can be used to contain 
    any scalar value type. It makes use of std::strstream to get and set values. This
    should be changed to std::stringstream when that class is available in g++.
*/

template <class ValueType>
class eoValueParam : public eoParam
{
public :
    eoValueParam(void) : eoParam() {}

  /**
   * Construct a Param.
   * @param _defaultValue       The default value
   * @param _longName           Long name of the argument
   * @param _description        Description of the parameter. What is useful for.
   * @param _shortName          Short name of the argument (Optional)
   * @param _required           If it is a necessary parameter or not
   */
    eoValueParam (ValueType _defaultValue, 
                std::string _longName, 
                std::string _description, 
                char _shortHand = 0,
                bool _required = false)
    : repValue(_defaultValue), eoParam(_longName, "", _description, _shortHand, _required)
    {
        eoParam::defValue(getValue());
    }
    
    ValueType& value()              { return repValue; }
    ValueType  value() const        { return repValue; }

    std::string getValue(void) const
    {
        std::ostrstream os;
        os << repValue;
        os << std::ends;
        return os.str(); 
    }
    
    void setValue(std::string _value)
    {
        std::istrstream is(_value.c_str());
        is >> repValue;
    }

private :
    ValueType repValue;
};

/*template <class ContainerType>
class eoContainerParam : public eoParam
{
public :
    eoContainerParam (ContainerType& value, string _shortName, string _longName, 
	            string _default, 
	            string _description, 
                bool _required,
	            bool _change )
    : value(_value), eoParam(_shortName, _longName, _description, _default, _required, _change)
    {}


  //  void setValue(const string & _value)
   // {
   //     std::istringstream is(_value);
    //    copy(istream_iterator<Container::value_type>(is), istream_iterator<Container::value_type>(), back_inserter(value));
   // }

private :
    ContainerType& value;
};*/


#endif
