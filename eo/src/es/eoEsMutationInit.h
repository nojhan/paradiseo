// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEsMutationInit.h
// (c) GeNeura Team, 1998 - EEAAX 1999 - Maarten Keijzer 2000
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoEsInit_h
#define _eoEsInit_h

#include <utils/eoParser.h>

/**
\ingroup EvolutionStrategies

    eoESMutationInit. Proxy class that is used for initializing the mutation
    operator. It provides an interface between eoEsMutate and the abstract
    parameterLoader. It also provides the names for the parameters in this 
    class as virtual protected member functions.
    
    If you have more than a single ES in a project that need different
    names in the configuration files, you might consider overriding this class 
    to change the names.

  @see eoEsMutate
*/
class eoEsMutationInit
{
  public :

    eoEsMutationInit(eoParameterLoader& _parser) : parser(_parser), TauLclParam(0), TauGlbParam(0), TauBetaParam(0) {}

    double TauLcl(void)
    {
        if (TauLclParam == 0)
        {
            TauLclParam = &parser.createParam(1.0, TauLclName(), "Local Tau", TauLclShort(), section());
        }

        return TauLclParam->value();
    }

    double TauGlb(void)
    {
        if (TauGlbParam == 0)
        {
            TauGlbParam = &parser.createParam(1.0, TauGlbName(), "Global Tau", TauGlbShort(), section());
        }

        return TauGlbParam->value();
    }

    double TauBeta(void)
    {
        if (TauBetaParam == 0)
        {
            TauBetaParam = &parser.createParam(0.0873, TauBetaName(), "Beta", TauBetaShort(), section());
        }

        return TauBetaParam->value();
    }

  protected :
    
    virtual std::string section(void) 
    { return "Parameters of ES mutation (before renormalization)"; }

    virtual std::string TauLclName(void) const       { return "TauLcL"; }
    virtual char   TauLclShort(void) const           { return 'l'; }
    
    virtual std::string TauGlbName(void) const      { return "TauGlb"; }
    virtual char   TauGlbShort(void) const          { return 'g'; }

    virtual std::string TauBetaName(void) const        { return "Beta"; }
    virtual char   TauBetaShort(void) const            { return 'b'; }

  private :

      eoParameterLoader& parser;

      eoValueParam<double>* TauLclParam;
      eoValueParam<double>* TauGlbParam;
      eoValueParam<double>* TauBetaParam;
};

#endif