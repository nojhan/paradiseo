// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSBXcross.h
// (c) Maarten Keijzer 2000 - Marc Schoenauer 2001
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#include <algorithm>    // swap_ranges
#include <utils/eoParser.h>
#include <utils/eoRNG.h>
#include <es/eoReal.h>
#include <utils/eoRealBounds.h>
#include <utils/eoRealVectorBounds.h>



/**
* @ingroup Real
* @ingroup Variators
*/
template<class EOT> class eoSBXCrossover: public eoQuadOp<EOT>
{
 public:
  /****
   * (Default) Constructor.
   * The bounds are initialized with the global object that says: no bounds.
   *
   *
   */
    eoSBXCrossover(const double& _eta = 1.0) :
    bounds(eoDummyVectorNoBounds), eta(_eta), range(1) {}


  //////////////////////////////////////////////

  /**
   * Constructor with bounds
   * @param _bounds an eoRealVectorBounds that contains the bounds
   * @param _eta the amount of exploration OUTSIDE the parents
   *               as in BLX-alpha notation (Eshelman and Schaffer)
   *               0 == contractive application
   *               Must be positive
   */



     eoSBXCrossover(eoRealVectorBounds & _bounds,
                     const double& _eta = 1.0) :
     bounds(_bounds), eta(_eta), range(1) {}

  ///////////////////////////////////////////////

  //////////////////////////////////////////////

  /**
   * Constructor from a parser. Will read from the argument parser
   * eoRealVectorBounds that contains the bounds
   * eta, the SBX parameter
   */

     eoSBXCrossover(eoParser & _parser) :
    // First, decide whether the objective variables are bounded
    // Warning, must be the same keywords than other possible objectBounds elsewhere
       bounds (_parser.getORcreateParam(eoDummyVectorNoBounds, "objectBounds", "Bounds for variables", 'B', "Variation Operators").value()) ,
    // then get eta value
       eta (_parser.getORcreateParam(1.0, "eta", "SBX eta parameter", '\0', "Variation Operators").value()) ,
       range(1) {}


  /// The class name.
  virtual std::string className() const { return "eoSBXCrossover"; }

  /*****************************************
   * SBX crossover - modifies both parents *
   * @param _eo1 The first parent          *
   * @param _eo2 The first parent          *
   *****************************************/
  bool operator()(EOT& _eo1, EOT& _eo2)
    {
      unsigned i;
      double r1, r2, beta;

        for (i=0; i<_eo1.size(); i++)
          {
      double u = rng.uniform(range) ;

      if ( u <= 0.5 )
        beta = exp( (1/(eta+1))*log(2*u));
      else
        beta = exp((1/(eta+1))*log(1/(2*(1-u))));



              r1=_eo1[i];
              r2=_eo2[i];
              _eo1[i] =0.5*((1+beta)*r1+(1-beta)*r2);
              _eo2[i] =0.5*((1-beta)*r1+(1+beta)*r2);


              if(!(bounds.isInBounds(i,_eo1[i])))
                bounds.foldsInBounds(i,_eo1[i]);
              if(!(bounds.isInBounds(i,_eo2[i])))
                bounds.foldsInBounds(i,_eo2[i]);



           }
        return true;
        }



protected:
  eoRealVectorBounds & bounds;
  double eta;
  double range;                    // == 1
};
