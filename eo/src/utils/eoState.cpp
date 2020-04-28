#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <algorithm>
#include <fstream>
#include <sstream>

#include "eoState.h"
#include "../eoObject.h"
#include "../eoPersistent.h"

// using namespace std;



void eoState::removeComment(std::string& str, std::string comment)
{
    std::string::size_type pos = str.find(comment);

    if (pos != std::string::npos)
    {
        str.erase(pos, str.size());
    }
}

bool eoState::is_section(const std::string& str, std::string& name)
{
    std::string::size_type pos = str.find(_tag_section_so);

    if (pos == std::string::npos)
        return false;
    //else

    std::string::size_type end = str.find(_tag_section_sc);

    if (end == std::string::npos)
        return false;
    // else

    // affect name, passed by reference
    // Note: substr( start, count )
    name = str.substr( pos + _tag_section_so.size(), end - _tag_section_so.size() );

    return true;
}

eoState::~eoState(void)
{
    for (unsigned i = 0; i < ownedObjects.size(); ++i)
    {
        delete ownedObjects[i];
    }
}

void eoState::registerObject(eoPersistent& registrant)
{
    std::string name = createObjectName(dynamic_cast<eoObject*>(&registrant));

    std::pair<ObjectMap::iterator,bool> res = objectMap.insert(make_pair(name, &registrant));

    if (res.second == true)
    {
        creationOrder.push_back(res.first);
    }
    else
    {
        throw eoException("Interval error: object already present in the state");
    }
}

void eoState::load(const std::string& _filename)
{
    std::ifstream is (_filename.c_str());

    if (!is)
    {
        // std::string str = "Could not open file " + _filename;
        throw eoFileError(_filename);
    }

    load(is);
}

// FIXME implement parsing and loading of other formats
void eoState::load(std::istream& is)
{
    std::string str;
    std::string name;

    getline(is, str);

    if (is.fail())
    {
        // std::string str = "Error while reading stream";
        throw eoFileError("stream");
    }

    while(! is.eof())
    { // parse section header
        if (is_section(str, name))
        {
            std::string fullString;
            ObjectMap::iterator it = objectMap.find(name);

            if (it == objectMap.end())
            { // ignore
                while (getline(is, str))
                {
                    if (is_section(str, name))
                        break;
                }
            }
            else
            {

                eoPersistent* object = it->second;

                // now we have the object, get lines, remove comments etc.

                std::string fullstring;

                while (getline(is, str))
                {
                  if (is.eof())
                    throw eoException("No section in load file");
                  if (is_section(str, name))
                    break;

                  removeComment(str, getCommentString());
                  fullstring += str + "\n";
                }
                std::istringstream the_stream(fullstring);
                object->readFrom(the_stream);
            }
        }
        else // if (is_section(str, name)) - what if file empty
          {
            getline(is, str);	// try next line!
            //      if (is.eof())
            //        throw runtime_error("No section in load file");
          }
    }

}

void eoState::save(const std::string& filename) const
{ // saves in order of insertion
    std::ofstream os(filename.c_str());

    if (!os)
    {
        // std::string msg = "Could not open file: " + filename + " for writing!";
        throw eoFileError(filename);
    }

    save(os);
}

//void eoState::save(std::ostream& os) const
//{ // saves in order of insertion
//    for (std::vector<ObjectMap::iterator>::const_iterator it = creationOrder.begin(); it != creationOrder.end(); ++it)
//    {
//        os << "\\section{" << (*it)->first << "}\n";
//        (*it)->second->printOn(os);
//        os << '\n';
//    }
//}

void eoState::saveSection( std::ostream& os, std::vector<ObjectMap::iterator>::const_iterator it) const
{
    os << _tag_section_so << (*it)->first << _tag_section_sc;

    os << _tag_content_s;
    (*it)->second->printOn(os);
    os << _tag_content_e;

    os << _tag_section_e;
}


void eoState::save(std::ostream& os) const
{
    os << _tag_state_so << _tag_state_name << _tag_state_sc;
   
    // save the first section
    assert( creationOrder.size() > 0 );
    // saves in order of insertion
    std::vector<ObjectMap::iterator>::const_iterator it = creationOrder.begin();
    saveSection(os,it);
    it++;

    while( it != creationOrder.end() ) {
        // add a separator only before [1,n] elements
        os << _tag_section_sep;
        saveSection(os, it);
        it++;
    }
    os << _tag_state_e;
}


std::string eoState::createObjectName(eoObject* obj)
{
    if (obj == 0)
    {
        std::ostringstream os;
        os << objectMap.size();
        return os.str();
    }
    // else

    std::string name = obj->className();
    ObjectMap::const_iterator it = objectMap.find(name);

    unsigned count = 1;
    while (it != objectMap.end())
    {
        std::ostringstream os;
        os << obj->className().c_str() << count++;
        name = os.str();
        it = objectMap.find(name);
    }

    return name;
}
