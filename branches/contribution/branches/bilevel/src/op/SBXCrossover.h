/*
* <SBXCrossover.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef SBXCROSSOVER_H_
#define SBXCROSSOVER_H_

#include <algorithm>    // swap_ranges
#include <utils/eoParser.h>
#include <utils/eoRNG.h>
#include <es/eoReal.h>
#include <utils/eoRealBounds.h>
#include <utils/eoRealVectorBounds.h>

template<class EOT> class SBXCrossover: public eoQuadOp<EOT>
{
public:
    /****
     * (Default) Constructor.
     * The bounds are initialized with the global object that says: no bounds.
     *
     *
     */
    SBXCrossover(const double& _eta = 1.0) :
            bounds(eoDummyVectorNoBounds), eta(_eta), range(1) {}


    //////////////////////////////////////////////

    /**
     * Constructor with bounds
     * @param _bounds an eoRealVectorBounds that contains the bounds
     * @param _alphaMin the amount of exploration OUTSIDE the parents
     *               as in BLX-alpha notation (Eshelman and Schaffer)
     *               0 == contractive application
     *               Must be positive
     */



    SBXCrossover(eoRealVectorBounds & _bounds,
                 const double& _eta = 1.0) :
            bounds(_bounds), eta(_eta), range(1) {}

    ///////////////////////////////////////////////

    //////////////////////////////////////////////

    /**
     * Constructor from a parser. Will read from the argument parser
     * eoRealVectorBounds that contains the bounds
     * eta, the SBX parameter
     */

    SBXCrossover(eoParser & _parser) :
            // First, decide whether the objective variables are bounded
            // Warning, must be the same keywords than other possible objectBounds elsewhere
            bounds (_parser.getORcreateParam(eoDummyVectorNoBounds, "objectBounds", "Bounds for variables", 'B', "Variation Operators").value()) ,
            // then get eta value
            eta (_parser.getORcreateParam(1.0, "eta", "SBX eta parameter", '\0', "Variation Operators").value()) ,
            range(1) {}

# define EPS 1.0e-14

    /// The class name.
    virtual std::string className() const {
        return "SBXCrossover";
    }

    /*****************************************
      * SBX crossover - modifies both parents *
      * @param _eo1 The first parent          *
      * @param _eo2 The first parent          *
      *****************************************/
    bool operator()(EOT& _eo1, EOT& _eo2)
    {
        unsigned i;
        double rand;
        double y1, y2, yl, yu;
        double c1, c2;
        double alpha, beta, betaq;
        bool changed = false;

        for (i=0; i<_eo1.size(); i++)
        {
            if (true)
            {
                if (fabs(_eo1[i] - _eo2[i]) > EPS) // pour éviter la division par 0
                {
                    // y2 doit être > à y1
                    if (_eo1[i] < _eo2[i])
                    {
                        y1 = _eo1[i];
                        y2 = _eo2[i];
                    }
                    else
                    {
                        y1 = _eo2[i];
                        y2 = _eo1[i];
                    }
                    yl = bounds.minimum(i);
                    yu = bounds.maximum(i);

                    rand = rng.uniform();

                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1));
                    alpha = 2.0 - pow( beta, -(eta + 1.0));
                    if (rand <= (1.0/alpha))
                    {
                        betaq = pow ( (rand * alpha), (1.0 / (eta + 1.0)));
                    }
                    else
                    {
                        betaq = pow ( (1.0 / (2.0 - rand * alpha)), (1.0 / (eta+1.0)));
                    }
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
                    alpha = 2.0 - pow( beta, -(eta + 1.0));
                    if (rand <= (1.0/alpha))
                    {
                        betaq = pow ( (rand * alpha), (1.0 / (eta + 1.0)));
                    }
                    else
                    {
                        betaq = pow ( (1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)));
                    }
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                    bounds.truncate(i, c1);
                    bounds.truncate(i, c2);

                    if (rng.flip())
                    {
                        _eo1[i] = c2;
                        _eo2[i] = c1;
                    }
                    else
                    {
                        _eo1[i] = c1;
                        _eo2[i] = c2;
                    }

                    changed = true;
                }
            }
        }

        return changed;
    }



protected:
    eoRealVectorBounds & bounds;
    double eta;
    double range;			   // == 1
};

#endif /*SBXCROSSOVER_H_*/
