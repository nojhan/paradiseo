#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iomanip> 

#include <utils/compatibility.h>

#include <utils/eoParser.h>

using namespace std;

void eoWarning(std::string str)
{
    cout << str << '\n';
}

std::ostream& printSectionHeader(std::ostream& os, std::string section)
{
    if (section == "")
        section = "General";

  os << '\n' << setw(10) << "######    " << setw(20) << section << setw(10) << "    ######\n";
  return os;
}

eoParameterLoader::~eoParameterLoader()
{
    for (int i = 0; i < ownedParams.size(); ++i)
    {
        delete ownedParams[i];
    }
}


eoParser::eoParser ( int _argc, char **_argv , string _programDescription, string _lFileParamName, char _shortHand) : 
    programName( _argv[0]),  
    programDescription( _programDescription), 
    parameterFile("", _lFileParamName, "Load using a configuration file", _shortHand),
    needHelp(false, "help", "Prints this message", 'h')
{
    strstream stream;
    
    for (int i = 1; i < _argc; ++i)
    {
        stream << _argv[i] << '\n';
    }

    readFrom(stream);

    processParam(parameterFile);
    processParam(needHelp);

    if (parameterFile.getValue() != parameterFile.defValue())
    {
        ifstream is (parameterFile.getValue().c_str());

        readFrom(is);
    }
}

void eoParser::processParam(eoParam& param, std::string section)
{
    doRegisterParam(param); // plainly register it
    params.insert(make_pair(section, &param));
}

void eoParser::doRegisterParam(eoParam& param) const
{
    if (param.required() && !isItThere(param))
    {
        throw std::runtime_error("required parameter missing");
    }

    pair<bool, string> value = getValue(param);

    if (value.first)
    {
        if (value.second == "") // it is there, but no value is given, default to "1"
            value.second = "1"; // for bool

        param.setValue(value.second);
    }
}

pair<bool, string> eoParser::getValue(eoParam& _param) const
{
    pair<bool, string> result(false, "");

    if (_param.shortName() != 0)
    {
        map<char, string>::const_iterator it = shortNameMap.find(_param.shortName());
        if (it != shortNameMap.end())
        {
            result.second = it->second;
            result.first = true;
            return result;
        }
    }

    map<string, string>::const_iterator it = longNameMap.find(_param.longName());

    if (it != longNameMap.end())
    {
        result.second = it->second;
        result.first = true;
        return result;
    }        
    // else (TODO: check environment, just long names)

    return result;
}

void eoParser::updateParameters() const
{
 typedef MultiMapType::const_iterator It;

  for (It p = params.begin(); p != params.end(); ++p)
  {
        doRegisterParam(*p->second);
  }
}

void eoParser::readFrom(istream& is)
{
  string str;
  
  while (is >> str)
  {
      if (str[0] == '#')
      { // skip the rest of the line
          string tempStr;
          getline(is, tempStr);
      }
      if (str[0] == '-')
      {
          if (str.size() < 2)
          {
              eoWarning("Missing parameter");
              needHelp.value() = true;
              return;
          }

          if (str[1] == '-') // two consecutive dashes
          {
              string::iterator equalLocation = find(str.begin() + 2, str.end(), '=');
              string value;
              
              if (equalLocation == str.end())
              { // TODO: it should be the next string
                value = "";
              }
              else
              {
                 value = string(equalLocation + 1, str.end());
              }

              string name(str.begin() + 2, equalLocation);
              longNameMap[name] = value;

          }
          else // it should be a char
          {
              string value(str.begin() + 2, str.end());
              shortNameMap[str[1]] = value;
          }
      }
  }

  updateParameters();
}

void eoParser::printOn(ostream& os) const
{
    typedef MultiMapType::const_iterator It;

    It p = params.begin();

    std::string section = p->first;

    printSectionHeader(os, section);

    //print every param with its value
    for (; p != params.end(); ++p) 
    {
        std::string newSection = p->first;

        if (newSection != section)
        {
            section = newSection;
            printSectionHeader(os, section);
        }
    
        eoParam* param = p->second;

        string str = "--" + param->longName() + "=" + param->getValue();
        
        os.setf(ios_base::left, ios_base::adjustfield);
        os << setw(40) << str;

        os << setw(0) << " # ";
        if (param->shortName())
            os << '-' << param->shortName() << " : ";
        os << param->description();
    
        if (param->required())
        {
            os << " REQUIRED ";
        }

        os  << '\n';
    }
}

void eoParser::printHelp(ostream& os) 
{   
    // print program name and description
    os << this->programName <<": "<< programDescription << "\n\n";

    // print the usage when calling the program from the command line
    os << "Usage: "<< programName<<" [Options]\n";
    // only short usage!
    os << "Options of the form \"-f[Value]\" or \"--Name[=value]\"" << endl; 

    os << "Where:"<<endl;

    typedef MultiMapType::const_iterator It;

    It p = params.begin();

    std::string section = p->first;

    printSectionHeader(os, section);

    //print every param with its value
    for (; p != params.end(); ++p) 
    {
        std::string newSection = p->first;

        if (newSection != section)
        {
            section = newSection;
            printSectionHeader(os, section);
        }

        if (p->second->shortName())
	        os << "-" << p->second->shortName() << ", ";
	    
        os << "--" <<p->second->longName() <<":\t"
	     << p->second->description() ;
	
	  os << "\n" << setw(20) << ( (p->second->required())?"Required":"Optional" );
      os <<". By default: "<<p->second->defValue() << '\n';
    } // for p

    os << '\n';

}
