/* (c) Thales group

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
    
*/

#ifndef EO_PARSER_BOOST_H
#define EO_PARSER_BOOST_H

#include "eoParam.h" 

#include <iostream>
#include <new> // std::bad_alloc()
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>
#include <typeinfo> // For operator typeid().

#include "eoLogger.h" // Required by eo::log (see eoParserBoost::printParsedOptions()).
#include "../eoObject.h"
//#include "../eoPersistent.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/variant.hpp>

namespace po = boost::program_options; 

typedef std::vector< po::basic_option<char> > opt_vect; // Used by eoParserBoost::printParsedOptions() ; in order to manipulate po::parsed_options easily.
typedef std::vector< std::basic_string<char> > str_vect; // Used by eoParserBoost::printParsedOptions() ; in order to manipulate po::parsed_options easily.

typedef boost::variant< int, 
                        short, 
                        long, 
                        unsigned int, 
                        unsigned short, 
                        unsigned long, 
                        char, 
                        float, 
                        double, 
                        long double, 
                        bool, 
                        char*,
                        std::string > my_variant; // For output writing (value cast) 
                                                  // and also for collecting all kind of information about the parameters (see "my_map")

typedef std::map< std::string, std::vector< my_variant > > my_map; // In order to save basic information about an option  
                                                                   // and collect them easily at the end for a complete data serialization
                                                                   // (see eoParserBoost::writeSettings()).
                                                                   // Key value: parameter longname ; mapped value : parameter shorthand, description and position. 

/**
    eoParserBoost: command line parser and configuration file reader/writer
    This class is inspired by eoParser. Contrary to eoParser 
    this class respects the POSIX/GNU standards.
    
    Dependencies : this class uses the Boost Program Options library :
                   $> sudo apt-get install libboost-dev-all
                   this class also requires c++11 : compile with flag -std=gnu++11
*/
class eoParserBoost: public eoObject//, public eoPersistent
{

public:

  /**
   * Constructor.
   * A complete constructor that reads the command-line arguments.
   * A configuration file can also be given for treatement, but it has to 
   * be specified as follows : 
   * $> ./myEo @param.rc            will then load using the parameter file param.rc
   *
   * @param _argc                   command line arguments count
   * @param _argv                   command line parameters
   * @param _programDescription     description of the work the program does
   */
    eoParserBoost(unsigned _argc, char **_argv , std::string _programDescription = "") throw(std::bad_alloc);

  /**
   * Destructor.
   */
    ~eoParserBoost();

    /**
    * Construct a option and set its default OR implicit value.
    *
    * @param _defaultValue       The value (default or implicit value)
    * @param _longName           Long name of the argument
    * @param _description        Description of the parameter. What is useful for.
    * @param _shortHand          Short name of the argument (Optional). 
    *                            By convention only one char (the 1st) will be considered.
    * @param _section            Name of the section to which the parameter belongs.
    *                            Allows to improve the output file writing but also to organize
    *                            the help indications which are printed on the terminal (--help).
    * @param _implicit           Tells if the value is a default value or an implicit value.
    * @param _position           The option POSITION
    */
    template <class ValueType>
    ValueType createParam(ValueType         _value,
                          const std::string _longName,
                          const std::string _description,
                          const std::string _shortHand,
                          const std::string _section,
                          bool              _implicit, 
                          int               _position = -1); 

   /**
    * Construct a option and set its default AND implicit value.
    *
    * @param _values             The combinaison of the default and implicit values.
    *                            The first member is the default value, the second is the implicit value.
    * @param _longName           Long name of the argument
    * @param _description        Description of the parameter. What is useful for.
    * @param _shortHand          Short name of the argument (Optional). 
    *                            By convention only one char (the 1st) will be considered.
    * @param _section            Name of the section to which the parameter belongs.
    *                            Allows to improve the output file writing but also to organize
    *                            the help indications printed on the terminal (--help).
    * @param _position           The option POSITION
    */
    template <class ValueType>
    std::pair< ValueType, ValueType > createParam(std::pair< ValueType, ValueType > _values,
                                                  const std::string                 _longName,
                                                  const std::string                 _description,
                                                  const std::string                 _shortHand,
                                                  const std::string                 _section, 
                                                  int                               _position = -1); 

   /**
    * Construct a parameter whose value has to be given in command-line 
    * (or from a configuration file).
    *
    * @param _longName           Long name of the argument
    * @param _description        Description of the parameter. What is useful for.
    * @param _shortHand          Short name of the argument (Optional). 
    *                            By convention only one char (the 1st) will be considered.
    * @param _section            Name of the section to which the parameter belongs.
    *                            Allows to improve the output file writing but also to organize
    *                            the help indications which are printed on the terminal (--help).
    * @param _position           The option POSITION
    */
    template <class ValueType>
    void createParam(const std::string _longName,
                     const std::string _description,
                     const std::string _shortHand, 
                     const std::string _section,
                     int               _position = -1); 

   /**
    * Process all the parameter values (parameters created from the source code, 
    * and modified in the command line or by means of a configuration file).
    * 
    * @param _configureHelpOption     Allows to configure automatically the help option 
    *                                 using the command-line specification --help or -h.           
    */
    void processParams(bool _configureHelpOption);

   /**
    * Cast the parameter value and copy it into the argument given as a reference.
    */
    template <class ValueType>
    void getValue(const std::string _longName, ValueType& _val);

   /**
    * Cast the parameter value as given by the ValueType type.
    */
    template <class ValueType>
    ValueType getValue(const std::string _longName);

   /**
    * Write and save processed settings in a file (data serialization).
    * Motivation: save settings given in command-line + configuration file
    * N.B. This output file can also be reused as configuration file for the next program execution.   
    */
    void writeSettings(const std::string _myFile);

    std::string className(void) const { return "eoParserBoost"; }

protected:

    std::string programName;
    std::string programDescription;

    char* configFile_name; // In order to register the configuration file name (given in command line). 
                           // N.B. The 'char*' type is required type by the Boost parser function.
    unsigned nb_args; // In order to register argc (given in command line). 
    char** args; // In order to register argv (given in command line). 
                 // N.B. The 'char**' type is required type by the Boost parser function.

    // The options decription component 
    // Describes the allowed options and what to do with the option values 
    po::options_description desc;

    // The positional options description component 
    po::positional_options_description pod;                                                  

    // The storage component            
    // Used to store all parameters that are processed
    po::variables_map vm; 

    // Contains all the options descriptions related to their own section. 
    // Key value : the section ; mapped value : the po::options_description object.
    // For the moment: tree depth level = 2.
    std::map< std::string, po::options_description > sections_map; 

    // In order to save basic information (section, short- and long- names and description) that we can't access easily 
    // with po::options_description or with po::positional_options_description.
    // Purpose: enable the complete data serialization by saving these settings in an output file (see function writeSettings()).
    // Key value : section ; mapped value : information about the parameter (parameter longname, shorthand, description, and position).
    std::map< std::string, my_map > info_map;

    // The help message for positional options has to be manually written.
    std::stringstream positional_options_help_message;

   /**
    * Throws an exception if the parameter longname already exists.
    */
    std::string checkIfExists(const std::string _longName, const std::string _shortHand);

   /**
    * Add a new key to the map related to the new section.
    */
    void manageSectionsMaps(const std::string _section);

   /**
    * Add a positional parameter into the positional options description.
    */
    void addPositionalOption(const std::string _longName, const std::string _shortHand, unsigned _position);

   /**
    * Save basic information about an parameter.
    */
    void saveInfo(const std::string _longName, 
                  const std::string _shortHand, 
                  const std::string _description, 
                  const std::string _section, 
                  int               _position);

   /**
    * Parse the argument values which are contained into the configuration file.
    */
    po::parsed_options parseConfigFile();

   /**
    * Print the parsed options - test display function 
    * In order to test the Boost parsing treatement. 
    */
    void printParsedOptions(po::parsed_options& _parsedOptions);

   /**
    * Print information about all the processed parameters.
    * This is the called function by the command-line option '--help' or '-h'.
    */
    void printHelp();

   /**
    * Return the right value cast.
    */
    std::pair<bool, my_variant> valueCast(const boost::any& _val);

   /**
    * Print settings on an output stream.
    */
    void writeSettings(std::ofstream& ofs);
};

#endif //  EO_PARSER_BOOST_H