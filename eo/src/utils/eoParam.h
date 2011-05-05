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

#include <cmath>
#include <iterator>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <eoScalarFitness.h>

/** @defgroup Parameters Parameters management
 *
 * A parameter is basically an object that stores a value and that can read/print it from/on streams.
 *
 * It is mainly used for command-line options (see eoParser) and eoStat.
 *
 * @ingroup Utilities
 * @{
 */

/**
   eoParam: Base class for monitoring and parsing parameters
*/
class eoParam
{
public:

    /** Empty constructor - called from outside any parser */
    eoParam ()
        : repLongName(""), repDefault(""), repDescription(""),
          repShortHand(0), repRequired(false)
        {}

    /** Construct a Param.
     *
     * @param _longName      Long name of the argument
     * @param _default       The default value
     * @param _description   Description of the parameter. What is useful for.
     * @param _shortName     Short name of the argument (Optional)
     * @param _required      If it is a necessary parameter or not
     */
    eoParam (std::string _longName, std::string _default,
             std::string _description, char _shortName = 0, bool _required = false)
        : repLongName(_longName), repDefault(_default),
          repDescription(_description ),
          repShortHand(_shortName), repRequired( _required)
        {}

    /**
     * Virtual destructor is needed.
     */
    virtual ~eoParam () {}

    /**
     * Pure virtual function to get the value out.
     */
    virtual std::string getValue () const = 0;

    /**
     * Pure virtual function to set the value
     */
    virtual void setValue(const std::string& _value)   = 0 ;

    /**
     * Returns the short name.
     */
    char shortName() const { return repShortHand; };

    /**
     * Returns the long name.
     */
    const std::string& longName() const { return repLongName; };

    /**
     * Returns the description of the argument
     */
    const std::string& description() const { return repDescription; };

    /**
     * Returns the default value of the argument
     */
    const std::string& defValue() const { return repDefault; };

    /**
     * Sets the default value of the argument,
     */
    void defValue( const std::string& str ) { repDefault = str; };

    /**
     * ALlows to change the name (see the prefix in eoParser.h)
     */
    void setLongName(std::string _longName) { repLongName = _longName;}

    /**
     * Returns if required or not.
     */
    bool required() const { return repRequired; };

private:
    std::string repLongName;
    std::string repDefault;
    std::string repDescription;
    char repShortHand;
    bool repRequired;
};



/**
   eoValueParam<ValueType>: templatized derivation of eoParam. Can be used to contain
   any scalar value type. It makes use of std::strstream to get and set values.

   @todo This should be changed to std::stringstream when that class is available in g++.

   Note also that there is a template specialization for std::pair<double, double> and
   for std::vector<double>. These stream their contents delimited with whitespace.
*/
template <class ValueType>
class eoValueParam : public eoParam
{
public :

    /** Construct a Param. */
    eoValueParam(void) : eoParam() {}

    /** Construct a Param.
     *
     * @param _defaultValue       The default value
     * @param _longName           Long name of the argument
     * @param _description        Description of the parameter. What is useful for.
     * @param _shortHand          Short name of the argument (Optional)
     * @param _required           If it is a necessary parameter or not
     */
    eoValueParam(ValueType _defaultValue,
                 std::string _longName,
                 std::string _description = "No description",
                 char _shortHand = 0,
                 bool _required = false)
        : eoParam(_longName, "", _description, _shortHand, _required),
          repValue(_defaultValue)
        {
            eoParam::defValue(getValue());
        }

    /** Get a reference on the parameter value

    @return parameter value
    */
    ValueType& value() { return repValue; }

    /** Get a const reference on the parameter value

    @overload

    @return parameter value
    */
    const ValueType& value() const { return repValue; }


    /** Change the parameter value
     */
    void value( ValueType val )
    {
        // convert to string
        std::ostringstream os;
        os << val;

        // convert to ValueType
        std::istringstream is( os.str() );
        is >> repValue;
    }


    /** Get the string representation of the value
     */
    std::string getValue(void) const
    {
        std::ostringstream os;
        os << repValue;
        return os.str();
    }


    /** @brief Set the value according to the speciied string

    For scalar types the textual represenation is typically quite
    straigtforward.

    For vector<double> we expect a list of numbers, where the first is
    an unsigned integer taken as the length ot the vector and then
    successively the vector elements. Vector elements can be separated
    by ',', ';', or ' '. Note, however, that eoParser does not deal
    correctly with parameter values contianing spaces (' ').

    @param _value Textual representation of the new value
    */
    void setValue(const std::string& _value)
    {
        std::istringstream is(_value);
        is >> repValue;
    }

protected:

    ValueType repValue;
};

/*
  Specialization for std::string
*/
template <>
inline std::string eoValueParam<std::string>::getValue() const
{
    return repValue;
}


template <>
inline void eoValueParam<bool>::setValue(const std::string& _value)
{
    if (_value.empty())
    {
        repValue = true;
        return;
    }
    std::istringstream is(_value);
    is >> repValue;
}


/// Because MSVC does not support partial specialization, the std::pair is a double, not a T
template <>
inline std::string eoValueParam<std::pair<double, double> >::getValue(void) const
{
    // use own buffer as MSVC's buffer leaks!
    std::ostringstream os;
    os << repValue.first << ' ' << repValue.second;
    return os.str();
}

/// Because MSVC does not support partial specialization, the std::pair is a double, not a T
template <>
inline void eoValueParam<std::pair<double, double> >::setValue(const std::string& _value)
{
    std::istringstream is(_value);
    is >> repValue.first;
    is >> repValue.second;
}

// The std::vector<std::vector<double> >
//////////////////////////////////
/// Because MSVC does not support partial specialization, the std::vector is a std::vector of doubles, not a T
template <>
inline std::string eoValueParam<std::vector<std::vector<double> > >::getValue(void) const
{
    std::ostringstream os;
    os << repValue.size() << ' ';
    for (unsigned i = 0; i < repValue.size(); ++i)
    {
        os << repValue[i].size() << ' ';
        std::copy(repValue[i].begin(), repValue[i].end(), std::ostream_iterator<double>(os, " "));
    }
    return os.str();
}

/// Because MSVC does not support partial specialization, the std::vector is a std::vector of doubles, not a T
template <>
inline void eoValueParam<std::vector<std::vector<double> > >::setValue(const std::string& _value)
{
    std::istringstream is(_value);
    unsigned i,j,sz;
    is >> sz;
    repValue.resize(sz);

    for (i = 0; i < repValue.size(); ++i)
    {
        unsigned sz2;
        is >> sz2;
        repValue[i].resize(sz2);
        for (j = 0; j < sz2; ++j)
        {
            is >> repValue[i][j];
        }
    }
}

// The std::vector<double>
//////////////////////////////////
/// Because MSVC does not support partial specialization, the std::vector is a double, not a T
template <>
inline std::string eoValueParam<std::vector<double> >::getValue(void) const
{
    std::ostringstream os;
    os << repValue.size() << ' ';
    std::copy(repValue.begin(), repValue.end(), std::ostream_iterator<double>(os, " "));
    return os.str();
}

/// Because MSVC does not support partial specialization, the std::vector is a double, not a T
template <>
inline void eoValueParam<std::vector<double> >::setValue(const std::string& _value)
{
    static const std::string delimiter(",;");
    std::istringstream is(_value);
    unsigned sz;
    is >> sz;
    repValue.resize(sz);
    for(unsigned i=0; i<repValue.size(); ++i) {
        char c;
        do {
            is >> c;
        } while((std::string::npos != delimiter.find(c)) && (! is.eof()));
        is >> repValue[i];
    }
}

// The std::vector<eoMinimizingFitness>
//////////////////////////////////
/// Because MSVC does not support partial specialization, the std::vector is a eoMinimizingFitness, not a T
template <>
inline std::string eoValueParam<std::vector<eoMinimizingFitness> >::getValue(void) const
{
    std::ostringstream os;
    os << repValue.size() << ' ';
    std::copy(repValue.begin(), repValue.end(), std::ostream_iterator<eoMinimizingFitness>(os, " "));
    return os.str();
}

/// Because MSVC does not support partial specialization, the std::vector is a eoMinimizingFitness, not a T
// NOTE: g++ doesn support it either!!!
template <>
inline void eoValueParam<std::vector<eoMinimizingFitness> >::setValue(const std::string& _value)
{
    std::istringstream is(_value);
    unsigned sz;
    is >> sz;
    repValue.resize(sz);
    std::copy(std::istream_iterator<eoMinimizingFitness>(is), std::istream_iterator<eoMinimizingFitness>(), repValue.begin());
}

// The std::vector<const EOT*>
//////////////////////////////////
template <>
inline std::string eoValueParam<std::vector<void*> >::getValue(void) const
{
    throw std::runtime_error("I cannot getValue for a std::vector<EOT*>");
    return std::string("");
}

template <>
inline void eoValueParam<std::vector<void*> >::setValue(const std::string&)
{
    throw std::runtime_error("I cannot setValue for a std::vector<EOT*>");
    return;
}

/*template <class ContainerType>
  class eoContainerParam : public eoParam
  {
  public :
  eoContainerParam (ContainerType& value, std::string _shortName, std::string _longName,
  std::string _default,
  std::string _description,
  bool _required,
  bool _change )
  : value(_value), eoParam(_shortName, _longName, _description, _default, _required, _change)
  {}


  //  void setValue(const std::string & _value)
  // {
  //     std::istd::stringstream is(_value);
  //    copy(std::istream_iterator<Container::value_type>(is), std::istream_iterator<Container::value_type>(), back_inserter(value));
  // }

  private :
  ContainerType& value;
  };*/

/**
 * Another helper class for parsing parameters like
 * Keyword(arg1, arg2, ...)
 *
 * It is basically a std::pair<std::string,std::vector<std::string> >
 *       first std::string is keyword
 *       the std::vector<std::string> contains all arguments (as std::strings)
 * See make_algo.h
 */

class eoParamParamType : public std::pair<std::string,std::vector<std::string> >
{
public:
    eoParamParamType(std::string _value)
        {
            readFrom(_value);
        }

    std::ostream & printOn(std::ostream & _os) const
        {
            _os << first;
            unsigned narg = second.size();
            if (!narg)
                return _os;

            // Here, we do have args
            _os << "(";
            if (narg == 1)         // 1 arg only
            {
                _os << second[0] << ")" ;
                return _os;
            }
            // and here more than 1 arg
            for (unsigned i=0; i<narg-1; i++)
                _os << second[i] << "," ;
            _os << second[narg-1] << ")";
            return _os;
        }

    std::istream & readFrom(std::istream & _is)
        {
            std::string value;
            _is >> value;
            readFrom(value);
            return _is;
        }

    void readFrom(std::string &  _value)
        {
            second.resize(0);              // just in case
            size_t pos = _value.find('(');
            if (pos >= _value.size())      // no arguments
            {
                first = _value;
                return;
            }
            // so here we do have arguments
            std::string t = _value.substr(pos+1);// the arguments
            _value.resize(pos);
            first = _value;    // done for the keyword (NOTE: may be empty std::string!)

            // now all arguments
            std::string delim(" (),");
            while ( (pos=t.find_first_not_of(delim)) < t.size())
            {
                size_t posEnd = t.find_first_of(delim, pos);
                std::string u = t.substr(pos,posEnd);//(t, pos);
                /*u.resize(posEnd - pos);*/
                second.push_back(u);
                t = t.substr(posEnd+1);
            }
        }
};

// at the moment, the following are defined in eoParser.cpp
std::ostream & operator<<(std::ostream & _os, const eoParamParamType & _rate);
std::istream & operator>>(std::istream & _is,  eoParamParamType & _rate);

/** @} */
#endif
