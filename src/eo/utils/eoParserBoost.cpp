//-----------------------------------------------------------------------------
// eoParserBoost.cpp
// (c) Thales group
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

Contact: http://eodev.sourceforge.net
Authors:

 */
//-----------------------------------------------------------------------------

#ifdef _MSC_VER
    #pragma warning(disable:4786) // to avoid long name warnings
#endif

#include "eoParserBoost.h"

eoParserBoost::eoParserBoost(unsigned _argc, char ** _argv, std::string _programDescription) throw(std::bad_alloc):
programName(_argv[0]),
programDescription(_programDescription),
nb_args(_argc),
desc("Allowed options") // Root node of the sections tree.
                        // The child nodes will been saved into the "sections_map".
                        // For the moment the tree depth level == 2.
{
    unsigned i, j, k, l;
    for (i = 1; i < _argc; i++)
    { // First check if a configuration file name (specified by symbol '@') has been given by command-line (e.g. @myFile.param).
        if ('@' == _argv[i][0])
        { // Copy this file name (in order to process it later). 
            j = strlen(_argv[i]+1) + 1;
            configFile_name = new char[j]; 
            strcpy(configFile_name, _argv[i]+1); 
            --nb_args;            
            break; // If several configuration file names have been given only the first one will be treated. As a consequence the others will be ignored.
        }
        else
            configFile_name = NULL;
    }
    args = new char*[nb_args]; 
    try
    {
        k = 0;
        for (i = 0; i < _argc; i++)
        { // Then copy all the arguments which have been given by command-line (in order to process them later) except for the configuration file name (already copied).
            if ('@' != _argv[i][0])
            {
                j = strlen(_argv[i]) + 1;
                args[k] = new char[j];
                strcpy(args[k++], _argv[i]);
            }
        }
    }
    catch (...)
    { // Clean up any allocated memory because the destructor will never be called.
        for (l = 0; l < i; l++)
            delete[] args[l];
        delete[] args; // Safe: assert(nb_args > 0);

        throw std::bad_alloc(); // Translate any exception to bad_alloc().
    }
}

eoParserBoost::~eoParserBoost() 
{
    unsigned i;
    for (i = 0; i < nb_args; i++)  
        delete[] args[i];
    delete[] args; // Safe: assert(nb_args > 0);

    if (configFile_name != NULL)
        delete[] configFile_name;
}

std::string eoParserBoost::checkIfExists(const std::string _longName, const std::string _shortHand)
{
    if (_longName == "")
    {
        std::string msg = "Exception: blank parameter longname forbidden";
        throw std::invalid_argument(msg);
    }
    
    for (std::map<std::string, my_map>::iterator it = info_map.begin(); it != info_map.end(); it++) 
    {  // For all sections...
        my_map m_m = it->second;
        for (my_map::iterator i = m_m.begin(); i != m_m.end(); i++) 
        {  // ... check that the parameter longname doesn't already exist.
            if (i->first == _longName)
            {
                std::string msg = std::string("Exception: two parameters with the same longname is forbidden (see parameters called \"") + _longName + std::string("\").");
                throw std::invalid_argument(msg);
            }
        }
    }  

    std::stringstream ss;
    ss << _longName << "," << _shortHand; // By following the POSIX/GNU standards Boost will only consider the first char of the shorthand string given as an argument.
                                          // e.g. 'option_name, oxxxxxxx' will be registered as 'option_name, o' 
    return ss.str();
}

void eoParserBoost::manageSectionsMaps(const std::string _section)
{
    if (!sections_map.count(_section))
    { // Add a new node to the "sections_map" and an other one to the "info_map" (which contains basic information about the parameters).
        po::options_description o_d(_section);
        sections_map.insert({_section, o_d});
        my_map m_m;
        info_map.insert({_section, m_m});
    }                 
}    

void eoParserBoost::addPositionalOption(const std::string _longName, const std::string _shortHand, unsigned _position)
{
    // First check that this position is not already reserved.
    if (pod.name_for_position(_position) != "")
    {
        std::string msg = std::string("Exception: position already reserved (see parameters named \"") 
                        + pod.name_for_position(_position) 
                        + std::string("\" and \"") 
                        + _longName + std::string("\").");
        throw std::invalid_argument(msg);
    }

    // WARNING from Boost Tutorial : "The positional_options_description class only specifies translation from position to name,
    // and the option name should still be registered with an instance of the options_description class."
    pod.add(_longName.c_str(), _position);

    // N.B.: the help message for positional options has to be manually written.
    positional_options_help_message << "  -" << _shortHand 
                                    << " [ "  <<  _longName << " ] "
                                    << "position = " << _position << "\n";
}

void eoParserBoost::saveInfo(const std::string _longName, 
                             const std::string _shortHand, 
                             const std::string _description,
                             const std::string _section,
                             int               _position)
{
    info_map.find(_section)->second
            .insert({_longName, {_shortHand, _description, _position}});  
}  

template <class ValueType>
ValueType eoParserBoost::createParam(ValueType         _value,
                                     const std::string _longName,
                                     const std::string _description,
                                     const std::string _shortHand, 
                                     const std::string _section,
                                     bool              _implicit,
                                     int               _position) 
{
    std::string str = checkIfExists(_longName, _shortHand); 

    if (_position > 0)
        addPositionalOption(_longName, _shortHand, static_cast<unsigned>(_position));

    // Add the parameter to the right section into the "sections_map".
    // If the section name is empty, the parameter belongs to the miscellaneous section called "General".
    std::string section = _section;
    if (section == "") 
        section = "General"; 

    manageSectionsMaps(section); 

    if (_implicit)
        sections_map.find(section)->second
            .add_options()
                (str.c_str(), po::value<ValueType>()->implicit_value(_value), _description.c_str()); 
    else
        sections_map.find(section)->second
                    .add_options()
                        (str.c_str(), po::value<ValueType>()->default_value(_value), _description.c_str()); 

    // Save now the information about parameter shorthand and longname, parameter section, description and position 
    // that later we want to recover easily in order to save these settings in an output file.
    // (see function writeSettings())
    saveInfo(_longName, _shortHand, _description, section, _position);
   
    return _value; 
}

template <class ValueType>
std::pair< ValueType, ValueType > eoParserBoost::createParam(const std::pair< ValueType, ValueType > _values,
                                                             const std::string                       _longName,
                                                             const std::string                       _description,
                                                             const std::string                       _shortHand,
                                                             const std::string                       _section, 
                                                             int                                     _position)
{
    std::string str = checkIfExists(_longName, _shortHand); 

    if (_position > 0)
        addPositionalOption(_longName, _shortHand, static_cast<unsigned>(_position));

    std::string section = _section;
    if (section == "") 
        section = "General"; 

    manageSectionsMaps(section); 

    sections_map.find(section)->second
                .add_options()
                    (str.c_str(), po::value<ValueType>()->default_value(_values.first)->implicit_value(_values.second), _description.c_str()); 

    saveInfo(_longName, _shortHand, _description, section, _position);

    return _values;
}

template <class ValueType>
void eoParserBoost::createParam(const std::string _longName,
                                const std::string _description,
                                const std::string _shortHand, 
                                const std::string _section,
                                int               _position)
{
    std::string str = checkIfExists(_longName, _shortHand);

    if (_position > 0)
        addPositionalOption(_longName, _shortHand, static_cast<unsigned>(_position));

    std::string section = _section;
    if (section == "") 
        section = "General";

    manageSectionsMaps(_section); 

    sections_map.find(_section)->second
                .add_options()
                    (str.c_str(), po::value<ValueType>()->required(), _description.c_str()); 

    saveInfo(_longName, _shortHand, _description, section, -1);
}

void eoParserBoost::printParsedOptions(po::parsed_options& _parsedOptions) 
{
    std::stringstream msg;
    msg << "\nPrinting parsed_options:\n";

    unsigned n = 0; 
    for (opt_vect::iterator it = _parsedOptions.options.begin(); it != _parsedOptions.options.end(); it++)
    {
        ++n;
        po::basic_option<char>& opt = *it;
        msg << "string_key[string]: "    << opt.string_key << '\t'
            << "position_key[int]: "     << opt.position_key << '\t'
            << "unregistered[bool]: "    << opt.unregistered << '\t'
            << "case_insensitive[bool]:" << opt.case_insensitive << '\t';

        str_vect o_tokens = opt.original_tokens;
        for (str_vect::iterator i = o_tokens.begin(); i != o_tokens.end(); i++) 
        { 
            const char* c = (*i).c_str();
            msg << "original_tokens: " << c << '\t';
        }

        str_vect val = opt.value;
        for (str_vect::iterator j = val.begin(); j != val.end(); j++) 
        { 
            const char* cv = (*j).c_str();
            msg << "value: " << cv << '\t';
        }

        msg << '\n';
    }

    msg << "number of options: " << n << "\n\n";

    #ifndef NDEBUG
        eo::log << eo::setlevel(eo::debug) << msg.str(); // Reminder: this is only a test display function.  
    #endif
}

po::parsed_options eoParserBoost::parseConfigFile()
{
    std::ifstream ifs(configFile_name);
    ifs.peek();
    // Check if the file exists first: 
    if (!ifs)
    { 
        std::string msg = std::string("Could not open configuration file: ") + configFile_name;
        throw std::runtime_error(msg);
    }
    else 
    {
        std::string str;
        std::vector<std::string> arguments;
        while (ifs >> str)
        {
            while (str[0] == ' ')
                str = str.substr(1, str.length()); // Ignore blanks at the beginning of the string.
            if (str[0] == '#')
            { // This part has been commented so skip the rest of the line.
                std::string tempStr;
                getline(ifs, tempStr);
            }
            else
                arguments.push_back(str);
        }

        return po::command_line_parser(arguments)
                    .options(desc)
                    #ifdef ALLOW_UNENREGISTERED 
                    .allow_unregistered() // DANGER: allows unenregistered parameters!
                                          // As a consequence if the flag is set to true and unenregistered parameters are found,
                                          // these unenregistered parameters won't be noticed and the whole configuration file 
                                          // will still be treated without throwing exception.
                    #endif
                    .run(); 
    }
 }

void eoParserBoost::processParams(bool _configureHelpOption)
{
    if (_configureHelpOption)
        createParam(std::pair< bool, bool > (false, true), "help", "Print this message.", "h", ""); 
                                                                                 
    for (std::map< std::string, po::options_description >::iterator it = sections_map.begin(); it != sections_map.end(); it++)
        desc.add(it->second); // Add all derived sections to the root.

    try
    {
        if (configFile_name != NULL) // If a configuration file name has been given in command line...
        { // ...it has to be processed first if we want command-line to have highest priority.         
            po::parsed_options parsed_options_from_file = parseConfigFile();
            //printParsedOptions(parsed_options_from_file); // test display function
            po::store(parsed_options_from_file, vm);
        }

        po::parsed_options parsed_options_from_cmd = po::command_line_parser(nb_args, args)
                                                            .options(desc)
                                                            .positional(pod)                                
                                                            #ifdef ALLOW_UNENREGISTERED 
                                                            .allow_unregistered() // DANGER: allows unenregistered parameters!
                                                                                  // As a consequence if the flag is set to true and unenregistered parameters are found,
                                                                                  // these unenregistered parameters won't be noticed and the whole configuration file 
                                                                                  // will still be treated without throwing exception.
                                                            #endif
                                                            .run(); 
        //printParsedOptions(parsed_options_from_cmd); // test display function
        po::store(parsed_options_from_cmd, vm);

        po::notify(vm); 
    }
    catch (const po::error &e)
    { 
        if (vm.count("help") && (vm["help"].as<bool>()))
        { // Display help if requested (by command-line) with '--help' or '-h'.
            std::cout << "-help specified" << std::endl;

            printHelp(); // Print help before throwing exception !
        }
        throw; // Throw exception even if help is requested because it can tell a little bit more to the user. 
               // But do not exit() : destructors won't be called! It has to be done directly into the source code of the program (see test file t-eoParserBoost.cpp). 
    }
    if (vm.count("help") && (vm["help"].as<bool>()))
    { // Display help if requested (by command-line) with '--help' or '-h'.
        std::cout << "-help specified" << std::endl;

        printHelp();
    }
}

void eoParserBoost::printHelp()
{
    std::cout << programName << std::endl;
    std::cout << programDescription << std::endl;
    std::cout << desc << std::endl;
    std::string msg = positional_options_help_message.str();
    if (msg != "") 
        std::cout << "Positional options:\n" << msg << std::endl;
    std::cout << "@param_file     defines a file where the parameters are stored\n" << std::endl;

}

template <class ValueType>
void eoParserBoost::getValue(const std::string _longName, ValueType& _val)
{
    if (vm.count(_longName)) 
        _val = vm[_longName].as<ValueType>();
    else
    { // Throw an exception if the parameter doesn't exist or if it's an undeclared implicit parameter.
        std::string msg = _longName + std::string(" does not exist or this implicit parameter has not been declared (see getValue()).");
        throw std::invalid_argument(msg);
    }
}

template <class ValueType>
ValueType eoParserBoost::getValue(const std::string _longName)
{
    if (vm.count(_longName)) 
        return vm[_longName].as<ValueType>();
    std::string msg = _longName + std::string(" does not exist or this implicit parameter has not been declared (see getValue()).");
    throw std::invalid_argument(msg); // Throw an exception if the parameter doesn't exist or if it's an undeclared implicit parameter.
}

std::pair<bool, my_variant> eoParserBoost::valueCast(const boost::any& _val)
{
    if (_val.type() == typeid(int)) 
        return std::make_pair(true, boost::any_cast<int>(_val));
    if (_val.type() == typeid(short)) 
        return std::make_pair(true, boost::any_cast<short>(_val));
    if (_val.type() == typeid(long)) 
        return std::make_pair(true, boost::any_cast<long>(_val));
    if (_val.type() == typeid(unsigned int)) 
        return std::make_pair(true, boost::any_cast<unsigned int>(_val));   
    if (_val.type() == typeid(unsigned short)) 
        return std::make_pair(true, boost::any_cast<unsigned short>(_val));
    if (_val.type() == typeid(unsigned long)) 
        return std::make_pair(true, boost::any_cast<unsigned long>(_val));
    if (_val.type() == typeid(char)) 
        return std::make_pair(true, boost::any_cast<char>(_val));
    if (_val.type() == typeid(float))
        return std::make_pair(true, boost::any_cast<float>(_val));
    if (_val.type() == typeid(double)) 
        return std::make_pair(true, boost::any_cast<double>(_val));
    if (_val.type() == typeid(long double)) 
        return std::make_pair(true, boost::any_cast<long double>(_val));
    if (_val.type() == typeid(bool)) 
        return std::make_pair(true, boost::any_cast<bool>(_val)); 
    if (_val.type() == typeid(std::string)) 
        return std::make_pair(true, boost::any_cast<std::string>(_val)); 
    if (_val.type() == typeid(char*)) 
        return std::make_pair(true, boost::any_cast<char*>(_val));
    /*
    if (_val.type() == typeid(std::vector< int >))
         return std::make_pair(true, boost::any_cast< std::vector< int > >(_val));
    if (_val.type() == typeid(std::vector< short >))
        return std::make_pair(true, boost::any_cast< std::vector< short > >(_val));
    if (_val.type() == typeid(std::vector< long >))
        return std::make_pair(true, boost::any_cast< std::vector< long > >(_val));    
    if (_val.type() == typeid(std::vector< unsigned int >))
        return std::make_pair(true, boost::any_cast< std::vector< unsigned int > >(_val));
    if (_val.type() == typeid(std::vector< unsigned short >))
        return std::make_pair(true, boost::any_cast< std::vector< unsigned short > >(_val));
    if (_val.type() == typeid(std::vector< unsigned long >))
        return std::make_pair(true, boost::any_cast< std::vector< unsigned long > >(_val));
    if (_val.type() == typeid(std::vector< char >))
        return std::make_pair(true, boost::any_cast< std::vector< char > >(_val));
    if (_val.type() == typeid(std::vector< float >))
        return std::make_pair(true, boost::any_cast< std::vector< float > >(_val));
    if (_val.type() == typeid(std::vector< double >))
        return std::make_pair(true, boost::any_cast< std::vector< double > >(_val));
    if (_val.type() == typeid(std::vector< long double >))
        return std::make_pair(true, boost::any_cast< std::vector< long double > >(_val));
    if (_val.type() == typeid(std::vector< bool >))
        return std::make_pair(true, boost::any_cast< std::vector< bool > >(_val));
    if (_val.type() == typeid(std::vector< str::string >))
        return std::make_pair(true, boost::any_cast< std::vector< std::string > >(_val));
    if (_val.type() == typeid(std::vector< char* >))
        return std::make_pair(true, boost::any_cast< std::vector< char* > >(_val));
        */
    else
        return std::make_pair<bool, std::string>(false, "Unexpected type"); // Type is not referenced. Cast failed.
}

void eoParserBoost::writeSettings(std::ofstream& _ofs) 
{
    for (std::map<std::string, my_map>::iterator it = info_map.begin(); it != info_map.end(); it++) 
    {  // Print all parameters section by section.
        
        std::string section = it->first;
        _ofs << "####    " << section << "    ####\n"; 

        my_map m_m = it->second;
        for (my_map::iterator i = m_m.begin(); i != m_m.end(); i++) 
        {   
            std::string    longName = i->first;  
            std::string   shortHand = boost::get<std::string>((i->second)[0]);
            std::string description = boost::get<std::string>((i->second)[1]);
            int            position = boost::get<int>((i->second)[2]);
            
            const po::variable_value& var = vm.find(longName)->second; 

/*********************************************************************************************************************************
 * writing convention for the parameter value : commented or not commented ?                      
 *
 *          - default (commented == TRUE)
 *          |                         - not declared (commented == TRUE) WARNING: value has no type, cast will fail 
 *   value -|              - implicit | 
 *          |              |          + declared in command-line or in configuration file (commented == FALSE)
 *          + not default -|
 *                         |               
 *                         + not implicit => value has been changed by configuration file or in command-line (commented == FALSE)
 *
 *********************************************************************************************************************************/

            if ( var.defaulted() || (!var.defaulted() && !vm.count(longName)) )
                _ofs << "# --" << longName; 
            else
                _ofs << "--" << longName; 

            if (vm.count(longName)) 
            { // Case where it is not an undeclared implicit parameter, e.g. we are sure that the value has a type
                const boost::any& val = var.value(); 
                if (!val.empty()) 
                {
                    _ofs << "=";
                    _ofs << valueCast(val).second;
                }
            }

            if ( (shortHand != "") && (description == "") && (position <= 0) ) 
                _ofs << "\t" << "# -" << shortHand.substr(0,1); 
            if ( (shortHand != "") && (description != "") && (position <= 0) )
                _ofs << "\t" << "# -" << shortHand.substr(0,1) << ": " << description;
            if ( (shortHand != "") && (description != "") && (position > 0) )
                _ofs << "\t" << "# -" << shortHand.substr(0,1) << ": " << description << ", position: " << position;
            if ( (shortHand == "") && (description != "") && (position <= 0) )
                _ofs << "\t" << "# " << description;
            if ( (shortHand == "") && (description == "") && (position > 0) )
                _ofs << "\t" << "# position: " << position;;

            _ofs << "\n";
        } 
        _ofs << "\n";
    }
}

void eoParserBoost::writeSettings(const std::string _myFile) 
{
    std::ofstream ofs(_myFile);
    std::string msg; 
    // Check first if the output file exists :
    if (!ofs.is_open())
    {   
        msg = std::string("Could not open the output file: ") + _myFile;
        throw std::runtime_error(msg);
    }
    else 
    { 
        writeSettings(ofs);
        
        ofs.close();
        msg = std::string("Settings save in ") + _myFile;
    }
    #ifndef NDEBUG
        eo::log << eo::setlevel(eo::debug) << msg; 
    #endif
}
