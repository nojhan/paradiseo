// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
/* eoParser.h
  some classes to parser either the command line or a parameter file

 (c) Marc Schoenauer and Geneura team, 1999

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
 */
//-----------------------------------------------------------------------------

#ifndef _PARSER_H
#define _PARSER_H

#include <string.h> // for strcasecmp ... maybe there's a c++ way of doing it?
#ifdef _MSC_VER
#define strcasecmp(a,b) _strnicmp(a,b,strlen(a))
#endif

// STL includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <strstream>
#include <ctime>

//-----------------------------------------------------------------------------
// Class UExceptions - probably useless ???
//-----------------------------------------------------------------------------
/**
 * This class manages exceptions. It´s barely an extension of the standard except
ion, 
 * but it can be initialized with an STL string. Called UException (utils-except
ion)+
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


//-----------------------------------------------------------------------------
// Class Param
//-----------------------------------------------------------------------------

/**
 * A param repesents an argument that can be passed to a program in the command line
 */
class Param {
public:
  
  /**
   * Type of params
   */
  enum valueType { INT, UL, FLOAT, STRING, BOOL, ARRAY, TITLE };
  
  /**
   * Construct an Param.
   * @param _shortName     Short name of the argument
   * @param _longName      Long name of the argument
   * @param _default       The default value
   * @param _valueType     Type of the parameter ("integer","unsigned long", "float","char", "bool" and so on)
   * @param _description   Description of the parameter. What is useful for.
   * @param _required      If it is a necessary parameter or not
   */
  Param (string _shortName="-h", string _longName="--help", 
	 string _default = "", valueType _valType= STRING,
	 string _description="Shows this help", 
	 bool _required=false )
    : repShortName(_shortName), repLongName(_longName), 
    repDescription(_description ), repEnv(""), repDefault(_default),
      repValue(_default), repValType( _valType), 
      repRequired( _required), repChanged(false) {
    
    const char *c = repLongName.c_str();
    for(unsigned i=0; i<repLongName.length() ; i++,c++) {
      if( *c != '-' ) break;
    }
    //initialize "repEnv" depending on the long name of the parameter.
    //previously the "-" (if exist) are skiped.
    repEnv = c ;
  };
  
  /**
   * Copy constructor
   * @param _param The source param.
   */
  Param (const Param& _param) :
    repShortName(_param.repShortName), repLongName(_param.repLongName), 
    repDescription(_param.repDescription ), repEnv(_param.repEnv), 
    repDefault(_param.repDefault),
    repValue(_param.repValue), repValType(_param.repValType), 
    repRequired(_param.repRequired), repChanged(_param.repChanged) {};
  
  /**
   * Virtual destructor is needed.
   */
  virtual ~Param () {};
  
  /**
   * Returns the short name.
   */
  const string& shortName ( void ) const { return repShortName; };
  
  /**
   * Returns the long name.
   */
  const string& longName ( void ) const { return repLongName; };
  
  /**
   * Returns the description of the argument
   */
  const string& description ( void ) const { return repDescription; };
  
  /**
   * Returns the environment variable of the argument
   */
  const string& environment ( void ) const { return repEnv; };
  
  /**
   * Returns the default value of the argument
   */
  const string& defValue ( void ) const { return repDefault; };
  
  /**
   * Sets a value for the param.
   * @param _value  The new value.
   */
  void value ( const string& _value ) { repValue = _value; repChanged = true; };
  
  /**
   * Returns the value of the param.
   */
  const string& value ( void ) const { return repValue; };
  
  /**
   * Returns if required or not.
   */
  bool required ( void ) const { return repRequired; };
  
  /**
   * Returns the type of the param's value.
   */
  Param::valueType valType( void ) const { return repValType; };

  /**
   * Returns true if the default value of the param has changed.
   */
  bool changed( void ) const { return repChanged; };  
  
private:
  string repShortName;
  string repLongName;
  string repDescription;
  string repEnv;
  string repDefault;
  
  string repValue;
  Param::valueType repValType;
  bool repRequired;
  bool repChanged;
};

/// This operator is defined to avoid errors in some systems
inline bool operator < ( const Param& _p1, const Param& _p2 ) {
  return ( _p1.shortName() < _p2.shortName() );
}

/// This operator is defined to avoid errors in some systems
inline bool operator == ( const Param& _p1, const Param& _p2 ) {
  return ( _p1.shortName() == _p2.shortName() );
}

//-----------------------------------------------------------------------------
// Class Parser
//-----------------------------------------------------------------------------

/**
 * Parses the command line / input parameter file / environment variables.
 */
class Parser {
public:
  
  /**
   * Constructor 
   * @param _argc, _ argv        command line arguments
   * @param  _programDescription  Description of the work the program does
   */
  Parser ( int _argc, char **_argv , string _programDescription, 
	   string _sFileParamName = "-P",
	   string _lFileParamName = "--Param") : 
    params(), 
    programName( _argv[0]),  programDescription( _programDescription),
    parse_argc(_argc), parse_argv(_argv), InputFileName("") {

    // the input file name has to be read immediately - from command-line or environement (not input0file :-)
    string _default = _argv[0];
    _default += ".param";
    Param param (_sFileParamName, _lFileParamName, _default, Param::STRING, "Name of the input file", 0);

    // FIRST: look for the corresponding environment variable
      if( getenv( param.environment().c_str() ) ) 
	param.value(getenv(param.environment().c_str()) ); 

    // LAST (highest priority) parse the command line arguments 
    for (int i=1 ; i<parse_argc ; i++)
      if( ( ! strcasecmp(param.longName().c_str(), parse_argv[i]) ) ||
 	  ( ! strcmp(param.shortName().c_str(), parse_argv[i]) )
	  ) {			   // found the parameter name
	param.value(parse_argv[i+1]);
	break;
      }
    // Now try to open the file
    ifstream is(param.value().c_str());
    if (is)			   // file exists ???
      InputFileName = param.value().c_str();

    params.push_back( param );
  };
  
  /**
   * Copy constructor
   * @param _parser The source parser
   */
  Parser ( const Parser& _parser ) : 
    params(_parser.params),
    programName( _parser.programName), 
    programDescription(_parser.programDescription), 
    parse_argc(_parser.parse_argc),
    parse_argv(_parser.parse_argv),
    InputFileName(_parser.InputFileName)
  {};
  
  /**
   * Virtual destructor is needed.
   */
  virtual ~Parser () {};
  
  
  /**
   * Adds a fake parameter == title in the output file
   * @param the title
   */
  void AddTitle (const string& _title)  
  { 
    Param param ( "", "", "", Param::TITLE, _title, false ); 
    params.push_back( param ); 
  }     
  
  /**
   * Description of all parameter readings:
   * @param _shortName Short name of the param. 
   * @param _longName  Long name of the param. 
   * @param _default   Default value. 
   * @param _valType      Type of the parameter 
   * @param _description Parameter utility 
   * @param _required  If the parameter is necessary or not 
   */

  /**
   * Gets the string value of a param from the full parameter description
   * @param         see above
   */
  string getString (const string& _shortName, const string& _longName, 
		 const string& _default = "", 
		 const string& _description="", bool _required=false) {
    Param param ( _shortName, _longName, _default, Param::STRING, _description, _required );
    parse( param );
    params.push_back( param );

    return param.value();
  };
  
  /**
   * Gets the bool value of a param-flag from the full parameter description
   * @param         see above
   */
  
  bool getBool  (const string& _shortName, const string& _longName, 
		 const string& _description="") {
    Param param ( _shortName, _longName, "false", Param::BOOL, _description, false );
    parse( param );
    params.push_back( param );

    if (param.value() == "true") {
      return true;
    }
    else {
      return false;
    }
  };

  /**
   * Gets the "array" (vector of strings) value of a param from the full parameter description
   * @param         see above
   */
  vector<string> getArray  (const string& _shortName, const string& _longName, 
		 const string& _default = "", 
		 const string& _description="", bool _required=false) {
    Param param ( _shortName, _longName, _default, Param::ARRAY, _description, _required );
    parse( param );
    params.push_back( param );

    istrstream is(param.value().c_str());
    vector<string> retValue;
    string tmpStr;

    is >> tmpStr;
    while(is){
      retValue.push_back(tmpStr);
      is >> tmpStr;
    }
    return retValue;
  };
  
  /**
   * Gets the int value of a param given the full description of the parameter
   * @param         see above
   * @exception     BadType if the param's value isn't a correct int
   */

  int getInt  (const string& _shortName, const string& _longName, 
		 const string& _default = "", 
		 const string& _description="", bool _required=false) {
    Param param ( _shortName, _longName, _default, Param::INT, _description, _required );
    parse( param );
    params.push_back( param );

    // now gets the value
    istrstream is( param.value().c_str());
    int retValue;
    is >> retValue;
    
    if (!is) {
      throw Parser::BadType(param.longName().c_str(), param.value().c_str(), "float");
      return 0;
    } else {
      return retValue;
    }
  };

  /**
   * Gets the unsigned lon value of a param given ...
   * @param         see above
   * @exception     BadType if the param's value isn't a correct unsigned long
   */

  int getUnsignedLong  (const string& _shortName, const string& _longName, 
		 const string& _default = "", 
		 const string& _description="", bool _required=false) {
    Param param ( _shortName, _longName, _default, Param::UL, _description, _required );
    parse( param );
    params.push_back( param );

    // now gets the value
    istrstream is( param.value().c_str());
    unsigned long retValue;
    is >> retValue;
    
    if (!is) {
      throw Parser::BadType(param.longName().c_str(), param.value().c_str(), "float");
      return 0;
    } else {
      return retValue;
    }
  };

  /**
   * Gets the float value of a param given the  description of the parameter
   * @param         see above
   * @exception     BadType if the param's value isn't a correct int
   */

  float getFloat  (const string& _shortName, const string& _longName, 
		 const string& _default = "", 
		 const string& _description="", bool _required=false) {
    Param param ( _shortName, _longName, _default, Param::FLOAT, _description, _required );
    parse( param );
    params.push_back( param );

    // now gets the value
    istrstream is( param.value().c_str());
    float retValue;
    is >> retValue;
    
    if (!is) {
      throw Parser::BadType(param.longName().c_str(), param.value().c_str(), "float");
      return 0;
    } else {
      return retValue;
    }
  };


  string parse_string (istream & _is) {
    string paramValue;
    _is >> paramValue;
    //if the first character of the string or array is not a " => just one word or array-element.
    if( paramValue[0] != '\"' ) 
      return paramValue;

    if( paramValue[1] == '\"' ) // the empty string
      return "" ;

    //else => read until the next " (the end of the string).
    const char *c = paramValue.c_str();
    string tmpStr = c+1;// skip the "
    if (tmpStr[tmpStr.length()-1] == '\"') { // one word only
      tmpStr[tmpStr.length()-1] = '\0';
      return tmpStr;
    }

    bool stop = false;
    while (_is && !stop) {
      _is >> paramValue;
      // test last character of paramValue for "
      if (paramValue[paramValue.length()-1] == '\"') {
	paramValue[paramValue.length()-1] = '\0';
	stop = true;
      }
      tmpStr = tmpStr + " " + paramValue ;
    }
    return tmpStr;
  };


  void parse (Param & param) {
    int i;
    string tmpStr, ReadStr, FirstWord;

    // FIRST: look if the associated environment variables have any value, to use them.
      if( getenv( param.environment().c_str() ) ) {
	//cout <<"\t\t ENV param:  ,"<<p->shortName()<<",  ,"<<getenv(p->environment().c_str())<<endl;
	param.value(getenv(param.environment().c_str()) );
      }

    
    // SECOND: search the file parameter, if present
	  if ( InputFileName != "" ) {
		  ifstream is(InputFileName.c_str());
		  while (is) {
			  is >> tmpStr;
			  if (  ( !strcmp(tmpStr.c_str(), param.shortName().c_str()) ) ||
				  ( !strcasecmp(tmpStr.c_str(), param.longName().c_str()) )
				  ) {		   // found the keyword
				  
				  Param::valueType tmp = param.valType();
				  switch ( tmp ) {
				  case Param::TITLE:
					  cerr << "Error, we should not be there" << endl;
					  exit(1);
					  break;
				  case Param::BOOL : 
					  param.value("true" );
					  break;
					  
				  case Param::INT:  
				  case Param::UL:  
				  case Param::FLOAT: 
					  is >> tmpStr;
					  param.value(tmpStr);
					  break;
					  
				  case Param::STRING:
					  tmpStr = parse_string(is);
					  param.value(tmpStr);
					  break;
					  
				  case Param::ARRAY:
					  ReadStr = parse_string(is);
					  if ( ReadStr != "<" ) {  // no "<" ">" --> a single string in the array
						  param.value(ReadStr);
						  break;
					  }
					  // read next word - and keep it in case of <> mismatch
					  FirstWord = parse_string(is);
					  // test for empty array
					  if (FirstWord == ">") {
						  param.value("");
						  break;
					  }
					  // else, read all words until ">"
					  tmpStr = FirstWord;
					  ReadStr = parse_string(is);
					  while ( is && (ReadStr != ">") ) {
						  tmpStr = tmpStr + " " + ReadStr;
						  ReadStr = parse_string(is);
					  } 
					  
					  if (!is) {	   // there was a "<" without the corresponding ">"
						  throw Parser::BadArrayParam( param.longName(), FirstWord );
						  param.value(FirstWord); // assume unique string
					  }
					  else
						  param.value(tmpStr);
					  break;
				  }
			  }
		  }
	  }
		  
		  
		  // LAST (highest priority) parse the command line arguments
		  for (i=1 ; i<parse_argc ; i++)
			  if( ( ! strcasecmp(param.longName().c_str(), parse_argv[i]) ) ||
				  ( ! strcmp(param.shortName().c_str(), parse_argv[i]) )
				  ) {			   // found the parameter name
				  if (param.valType() == Param::BOOL) {
					  //cout <<"BOOL: "<<parse_argv[i]<<" <-- true"<<endl;
					  param.value("true");
				  }else{
					  if (param.valType() != Param::ARRAY) {  //only if it is not an array
						  //cout <<"TYPE: "<<parse_argv[i]<<" <-- "<<parse_argv[i+1]<<endl;
						  param.value(parse_argv[i+1]);
					  }else{                           //if it is an ARRAY
						  i++;
						  ReadStr = parse_argv[i++];
						  //cout <<"ARRAY: <--  ";
						  
						  if ( ReadStr != "<" ) {  // no "<" ">" --> a single string in the array
							  param.value(ReadStr);
						  }else{
							  // read next word - and keep it in case of <> mismatch
							  FirstWord = parse_argv[i++];
							  
							  // test for empty array
							  if (FirstWord == ">") {
								  param.value("");
							  }else{
								  // else, read all words until ">"
								  tmpStr = FirstWord;
								  ReadStr = parse_argv[i++];
								  while ( (i<parse_argc) && (ReadStr != ">") ) {
									  tmpStr = tmpStr + " " + ReadStr;
									  ReadStr = parse_argv[i++];
								  } 
								  //cout <<"tmpStr ;"<<tmpStr<<";   ("<<i<<","<<parse_argc<<") "<<endl;
								  
								  if ( (i>=parse_argc) && (ReadStr != ">") ) {	   // there was a "<" without the corresponding ">"
									  throw Parser::BadArrayParam( param.longName(), FirstWord );
									  param.value(FirstWord); // assume unique string
								  }else{
									  param.value(tmpStr);
								  }
							  }
						  }
					  }
				  }
				  break;
			  }
			  
			  //MS after trying all possibilities, and if the value has not changed 
			  // though the parameter was required, protest!
			  if (param.required() && !param.changed())
				  throw Parser::MissingReqParam(param.shortName());
			  
  };
  
  /**
   * Sets a new value for a param given its short name or its long name.
   * @param _name  One of the names of the param.
   * @param _value Value to be assigned.
   * @exception UnknownArg if the param doesn't exist
   * @exception MissingVal if the param hasn't got a value
   */
  Param::valueType setParamValue (const string& _name, const char* _value){
    vector<Param>::iterator pos;
    
    for (pos=params.begin() ; pos!=params.end() ; pos++)
      if (pos->shortName()==_name || pos->longName()==_name)
	break;
    
    // if found ...
    if (pos!=params.end()) {
      switch ( pos->valType() ) {
      case Param::TITLE:
	cerr << "Error, we should not be there" << endl;
	exit(1);
	break;
      case Param::BOOL :
	pos->value("true");
	break;
      case Param::ARRAY :
      case Param::INT: 
      case Param::UL: 
      case Param::FLOAT: 
      case Param::STRING: 
	if (_value != NULL){
	  pos->value(_value);
	}else{
	  throw Parser::MissingVal(_name);
	  return Param::BOOL;
	}
	break;
      } // switch
      
      return pos->valType();
      
    }else{
      throw Parser::UnknownArg(_name);
      return Param::BOOL;
    }
  };

  /// the output method - generate the .status file (unless other name is given)
  friend ostream & operator<< ( ostream & os, Parser & _parser )
  {
    vector<Param>::iterator p;     
    //print every param with its value
    for ( p=_parser.params.begin(); p!=_parser.params.end(); p++ ) {
      switch ( p->valType() ) {
      case Param::BOOL :
	if(p->value() == "true")
	  os << p->longName();
	else
	  os << "#" << p->longName() ; // so the name of the bool is commented out
	break;
	
      case Param::INT: 
      case Param::UL:  
      case Param::FLOAT: 
	os << p->longName()<<"  "<<p->value();
	break;
	
      case Param::ARRAY :
	os << p->longName() << "   < " << p->value().c_str() << " >" ;
	break;
      case Param::STRING: 
	os << p->longName()<<"   \""<<p->value().c_str()<<"\" ";
	break;
      case Param::TITLE:
	os << endl;	  // Title is in the description below
	break;
      } // switch
      os << "\t    #" << p->shortName() << " : " << p->description();
      if (p->valType() != Param::TITLE)
	os << " [" << p->defValue() << "]" ;
      os << endl;
    }
    return os;
  };
  
  /**
   * Prints out the list of parameters in the output file (if specified)
   */
  void outputParam(string _OutputFile="")
  {
    if (_OutputFile == "") {
      _OutputFile = parse_argv[0];
      _OutputFile += ".status";
    }
    
    ofstream os(_OutputFile.c_str()); 
    os << "Parameters used by \"" << programName << "\" ("
       << programDescription << ")" << endl << endl;
    os << *this;
  };
  
  /**
   * Prints an automatic help in the standard output using the information
   * provided by parameters
   */
  void printHelp() {
    vector<Param>::iterator p;
//    unsigned i;
    
    // print program name and description
    cout << this->programName <<": "<<programDescription<<endl<<endl;
    
    // print the usage when calling the program from the command line
    cout << "Usage: "<< programName<<" [Options]\n";
    // only short usage!
    cout << "Options of the form \"-ShortName value\" or \"--LongName value\"" << endl; 

//     for ( i=0,p=params.begin(); p!=params.end(); i++,p++ ) 
//       if( p->valType() != Param::TITLE ) {
// 	if( p->valType() != Param::BOOL ){
// 	  cout << ( (!p->required())?"[":"");
// 	  cout <<p->shortName()<<" value"<<i;
// 	  cout << ( (!p->required())?"]":"")<<" ";
// 	}else{
// 	  cout << "["<<p->shortName()<<"] ";
// 	}
//       } // for p
    cout << "Where:"<<endl;
    
    for ( p=params.begin(); p!=params.end(); p++ ) {
      if( p->valType() != Param::TITLE ) {
	cout << p->shortName()<<","<<p->longName()<<":\t"<<p->description()<<endl;

	cout << "\t(";
	switch ( p->valType() ) {
	case Param::INT: cout <<"Integer"; break;
	case Param::UL: cout <<"Unsigned Long Integer"; break;
	case Param::FLOAT: cout <<"Float"; break;
	case Param::STRING: cout <<"String"; break;
	case Param::ARRAY: cout <<"An array of strings, enclosed within < >"; break;
	case Param::BOOL: cout << "Flag"; break;
	case Param::TITLE: break;
	} // switch
	if(p->valType() == Param::BOOL)
	  cout << ") True if present" << endl;
	else
	  cout<<") "<<( (p->required())?"Required":"Optional" )<<". By default: "<<p->defValue()<<endl;
      } 
      else {
	cout << "\n\t    # " << p->description() << endl;
      }
    } // for p
    cout << endl;
  };
  
  /**
   * This class managges unknown argument exceptions.
   */
  class UnknownArg : public UException {
  public:
    
    /**
     * Constructor
     * @param _arg string to be shown when the exception occurs
     */
    UnknownArg( const string& _arg): UException( "Invalid argument: "+_arg ) { };
  };
  
  /**
   * This class managges bad param types.
   */
  class BadType : public UException {
  public:
    
    /**
     * Constructor
     * @param _param The param
     * @param _value The value of the param
     */
    BadType(const string& _param, const string& _value, const string& _correctType)
      : UException("The value '" + _value + "' assigned to the argument " + _param + " isn't a correct "+_correctType) { };
  };
  
  /**
   * This class managges exceptions produced when there isn't a value for a parameter.
   */
  class MissingVal : public UException {
  public:
    
    /**
     * Constructor
     * @param _param The param
     */
    MissingVal(const string& _param) : UException("Missing value for parameter " + _param) {};
  };
  
  /**
   * This class managges exceptions produced when the user forgot a required parameter.
   */
  class MissingReqParam : public UException {
  public:
    
    /**
     * Constructor
     * @param _shortName The param's short name
     */
    MissingReqParam(const string& _shortName) : UException("Missing required parameter " + _shortName) {};
  };
  
  /**
   * This class managges exceptions du to < without a > in array value
   */
  class BadArrayParam : public UException {
  public:
    
    /**
     * Constructor
     * @param _param The param
     * @param _first_word The first word read after the "<"
     */
    BadArrayParam(const string& _param, const string &_first_word) : 
      UException("Array parameter " + _param + ": No matching > ("  + _first_word 
		 + "... )") {};
  };
  
private:
  vector<Param> params;
  string programName; 
  string programDescription;
  int parse_argc;
  char **parse_argv;
  string InputFileName;
  
};

/// Reproducible random seed
// Maybe there is a better place for this subroutine (a separate .cpp?)
#include <eoRNG.h>

//----------------------------------
void InitRandom( Parser & parser) {
//----------------------------------
  unsigned long _seed;
  try {
    _seed = parser.getUnsignedLong("-S", "--seed", "0", "Seed for Random number generator" );
  }
  catch (UException & e)
    {
      cout << e.what() << endl;
      parser.printHelp();
      exit(1);
    }

  if (_seed == 0) {		   // use clock to get a "random" seed
   _seed = unsigned long( time( 0 ) );
   ostrstream s;
   s << _seed;
   parser.setParamValue("--seed", s.str());	   // so it will be printed out in the status file, and canbe later re-used to re-run EXACTLY the same run
  }
  rng.reseed(_seed);

  return;
}


#endif
