/*
* <PolynomialMutation.h>
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

#ifndef POLYNOMIALMUTATION_H_
#define POLYNOMIALMUTATION_H_

#include <utils/eoRealVectorBounds.h>

template<class EOT> class PolynomialMutation: public eoMonOp<EOT>
{
public:

    PolynomialMutation(eoRealVectorBounds & _bounds, const double& _p_mut = 0.5, const double& _eta = 1.0):
            p_mut(_p_mut), eta(_eta), bounds(_bounds) {}

    /// The class name.
    virtual std::string className() const {
        return "PolynomialMutation";
    }

    /**
      * Do it!
      * @param _eo The indi undergoing the mutation
      */
    bool operator()(EOT& _eo)
    {
        bool hasChanged=false;
        double rnd, delta1, delta2, mut_pow, deltaq, delta_max;
        double y, yl, yu, val, xy;

        for (unsigned j=0; j<_eo.size(); j++)
        {
            if (rng.flip(p_mut))
            {
                y = _eo[j];

                yl = bounds.minimum(j);
                yu = bounds.maximum(j);
                delta1 = (y-yl)/(yu-yl);
                delta2 = (yu-y)/(yu-yl);


                //Ajout
                if ( (y-yl) > (yu-y))
                    delta_max = delta2;
                else
                    delta_max= delta1;
                //fin ajout

                rnd = rng.uniform();
                mut_pow = 1.0/(eta+1.0);
                if (rnd <= 0.5)
                {
                    xy = 1.0-delta_max;//delta_max au lieu de delta1
                    val = 2.0*rnd+(1.0-2.0*rnd)*(pow(xy,(eta+1.0)));
                    deltaq =  pow(val,mut_pow) - 1.0;
                }
                else
                {
                    xy = 1.0-delta_max;//delta_max au lieu de delta2
                    val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(pow(xy,(eta+1.0)));
                    deltaq = 1.0 - (pow(val,mut_pow));
                }
                //ajout
                if (deltaq > delta_max)
                    deltaq = delta_max;
                else if (deltaq < -delta_max)
                    deltaq= -delta_max;
                //fin ajout
                y = y + deltaq*(yu-yl);

                bounds.truncate(j, y);
                _eo[j] = y;

                hasChanged = true;
            }
        }

        return hasChanged;
    }

private:
    double p_mut;
    double eta;
    eoRealVectorBounds & bounds;
};

#endif /*POLYNOMIALMUTATION_H_*/
