// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoState.h
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

#ifndef eoState_h
#define eoState_h

#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include <assert.h>

#include <eoFunctorStore.h>

class eoObject;
class eoPersistent;

/**
 eoState can be used to register derivants of eoPersistent. It will
 then in turn implement the persistence framework through members load
 and save, that will call readFrom and printOn for the registrated objects.

 It is derived from eoFunctorStore, so that it also serves as a place where
 all those nifty eo functors can be stored. This is useful in the case you
 want to use one of the make_* functions. These functions generally take as their
 last argument an eoFunctorStore (or a state) which is used to hold all dynamically
 generated data. Note however, that unlike with eoPersistent derived classes, eoFunctorBase
 derived classes are not saved or loaded. To govern the creation of functors,
 command-line parameters (which can be stored) are needed.

 @ingroup Utilities
*/
class eoState : public eoFunctorStore
{
public :

    eoState(std::string name="") :
      _tag_state_so(""),
      _tag_state_name(name),
      _tag_state_sc(""),
      _tag_section_so("\\section{"),
      _tag_section_sc("}\n"),
      _tag_content_s(""),
      _tag_content_e(""),
      _tag_section_sep(""),
      _tag_section_e("\n"),
      _tag_state_e("")
    {}

    ~eoState(void);

    void formatLatex(std::string name)
    {
        _tag_state_so = "";
        _tag_state_name = name;
        _tag_state_sc = "";
        _tag_section_so = "\\section{";
        _tag_section_sc = "}\n";
        _tag_content_s = "";
        _tag_content_e = "";
        _tag_section_sep = "";
        _tag_section_e = "\n";
        _tag_state_e = "";
    }

    void formatJSON(std::string name)
    {
        _tag_state_so = "{ \"";
        _tag_state_name = name;
        _tag_state_sc = "\":\n";
        _tag_section_so = "\t{ \"";
        _tag_section_sc = "\":\n";
        _tag_content_s = "\"";
        _tag_content_e = "\"";
        _tag_section_sep = ",\n";
        _tag_section_e = "\t}\n";
        _tag_state_e = "}\n";
    }


    /**
    * Object registration function, note that it does not take ownership!
    */
    void registerObject(eoPersistent& registrant);

    /**
    * Copies the object (MUST be derived from eoPersistent)
    * and returns a reference to the owned object.
    * Note: it does not register the object, this must be done afterwards!
    */
    template <class T>
    T&   takeOwnership(const T& persistent)
    {
        // If the compiler budges here, T is not a subclass of eoPersistent
        ownedObjects.push_back(new T(persistent));
        return static_cast<T&>(*ownedObjects.back());
    }

    /**
    * Loading error thrown when nothing seems to work.
    */
    struct loading_error : public std::runtime_error
    {
        loading_error(std::string huh = "Error while loading") : std::runtime_error(huh) {}
    };

    std::string getCommentString(void) const { return "#"; }

    /**
    * Reads the file specified
    *
    *   @param _filename    the name of the file to load from
    */
    void load(const std::string& _filename);

    /**
    * Reads the file specified
    *
    *   @param is    the stream to load from
    */
    void load(std::istream& is);

    /**
    * Saves the state in file specified
    *
    *   @param _filename    the name of the file to save into
    */
    void save(const std::string& _filename) const;

    /**
    * Saves the state in file specified
    *
    *   @param os       the stream to save into
    */
    void save(std::ostream& os) const;

private :
    std::string createObjectName(eoObject* obj);

    // first is Persistent, second is the raw data associated with it.
    typedef std::map<std::string, eoPersistent*> ObjectMap;

    ObjectMap objectMap;

    std::vector<ObjectMap::iterator> creationOrder;
    std::vector<eoPersistent*> ownedObjects;

    // private copy and assignment as eoState is supposed to be unique
    eoState(const eoState&);
    eoState& operator=(const eoState&);

    /* \@{
     * s=start, e=end
     * o=open, c=close
     *
     * { "my_state": 
     *     {
     *         "section_pop":"…",
     *         "section_rng":"…"
     *     }
     *  }
     *
     *                           // JSON         LATEX (default)
     */
    std::string _tag_state_so;   // { "          
    std::string _tag_state_name; // my_state     
    std::string _tag_state_sc;   // ":           

    std::string _tag_section_so; // { "          \\section{
    std::string _tag_section_sc; // ":           }\n

    std::string _tag_content_s; // "
    std::string _tag_content_e; // "

    std::string _tag_section_sep;// ,

    std::string _tag_section_e; // }           \n

    std::string _tag_state_e;   // }
    /** \@} */
 
    void removeComment( std::string& str, std::string comment);

    bool is_section(const std::string& str, std::string& name);

protected:
    void saveSection( std::ostream& os, std::vector<ObjectMap::iterator>::const_iterator it) const;

};
/** @example t-eoStateAndParser.cpp
 */

#endif
