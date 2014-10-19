// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; fill-column: 80; -*-

//-----------------------------------------------------------------------------
// eoESMute.h : ES mutation
// (c) Maarten Keijzer 2000 & GeNeura Team, 1998 for the EO part
//     Th. Baeck 1994 and EEAAX 1999 for the ES part
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
             marc.schoenauer@polytechnique.fr
                       http://eeaax.cmap.polytchnique.fr/
 */
//-----------------------------------------------------------------------------


#ifndef _EOESMUTATE_H
#define _EOESMUTATE_H

#include <cmath>
#include <eoInit.h>
#include <eoOp.h>
#include <es/eoEsMutationInit.h>
#include <es/eoEsSimple.h>
#include <es/eoEsStdev.h>
#include <es/eoEsFull.h>
#include <utils/eoRealBounds.h>
#include <utils/eoRNG.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/** ES-style mutation in the large

@ingroup Real
@ingroup Variators

Obviously, valid only for eoES*. It is currently valid for three types
of ES chromosomes:
- eoEsSimple:   Exactly one stdandard-deviation
- eoEsStdev:    As many standard deviations as object variables
- eoEsFull:     The whole guacemole: correlations, stdevs and object variables

Each of these three variant has it's own operator() in eoEsMutate and
intialization is also split into three cases (that share some commonalities)
*/
template <class EOT>
class eoEsMutate : public eoMonOp< EOT >
{
public:

    /** Fitness-type */
    typedef typename EOT::Fitness FitT;


    /** Initialization.

    @param _init Proxy class for initializating the three parameters
                 eoEsMutate needs
    @param _bounds Bounds for the objective variables
    */
    eoEsMutate(eoEsMutationInit& _init, eoRealVectorBounds& _bounds) : bounds(_bounds)
    {
        init(EOT(), _init); // initialize on actual type used
    }


    /** @brief Virtual Destructor */
    virtual ~eoEsMutate() {};


    /** Classname.

    Inherited from eoObject @see eoObject

    @return Name of class.
    */
    virtual std::string className() const {return "eoESMutate";};


    /** Mutate eoEsSimple

    @param _eo Individual to mutate.
    */
    virtual bool operator()( eoEsSimple<FitT>& _eo)
        {
            _eo.stdev *= exp(TauLcl * rng.normal());
            if (_eo.stdev < stdev_eps)
                _eo.stdev = stdev_eps;
            // now apply to all
            for (unsigned i = 0; i < _eo.size(); ++i)
            {
                _eo[i] += _eo.stdev * rng.normal();
            }
            bounds.foldsInBounds(_eo);
            return true;
        }


    /** Standard mutation in ES

    @overload

    Standard mutation of object variables and standard deviations in ESs.

    If there are fewer different standard deviations available than the
    dimension of the objective function requires, the last standard deviation is
    responsible for ALL remaining object variables.

    @param _eo Individual to mutate.

    @see
    Schwefel 1977: Numerische Optimierung von Computer-Modellen mittels der
    Evolutionsstrategie, pp. 165 ff.
    */
    virtual bool operator()( eoEsStdev<FitT>& _eo )
        {
            double global = TauGlb * rng.normal();
            for (unsigned i = 0; i < _eo.size(); i++)
            {
                double stdev = _eo.stdevs[i];
                stdev *= exp( global + TauLcl * rng.normal() );
                if (stdev < stdev_eps)
                    stdev = stdev_eps;
                _eo.stdevs[i] = stdev;
                _eo[i] += stdev * rng.normal();
            }
            bounds.foldsInBounds(_eo);
            return true;
        }


    /** Correlated mutations in ES

    @overload

    Mutation of object variables, standard deviations, and their correlations in
    ESs.

    @param _eo Individual to mutate.

    @see
    - H.-P. Schwefel: Internal Report of KFA Juelich, KFA-STE-IB-3/80, p. 43, 1980.
    - G. Rudolph: Globale Optimierung mit parallelen Evolutionsstrategien,
      Diploma Thesis, University of Dortmund, 1990.
    */
    virtual bool operator()( eoEsFull<FitT> & _eo )
    // Code originally from Thomas Bäck
        {
            // First: mutate standard deviations (as for eoEsStdev<FitT>).
            double global = TauGlb * rng.normal();
            unsigned i;
            for (i = 0; i < _eo.size(); i++)
            {
                double stdev = _eo.stdevs[i];
                stdev *= exp( global + TauLcl*rng.normal() );
                if (stdev < stdev_eps)
                    stdev = stdev_eps;
                _eo.stdevs[i] = stdev;
            }
            // Mutate rotation angles.
            for (i = 0; i < _eo.correlations.size(); i++)
            {
                _eo.correlations[i] += TauBeta * rng.normal();
                if ( fabs(_eo.correlations[i]) > M_PI )
                {
                    _eo.correlations[i] -= M_PI * (int) (_eo.correlations[i]/M_PI) ;
                }
            }
            // Perform correlated mutations.
            unsigned k, n1, n2;
            double d1,d2, S, C;
            std::vector<double> VarStp(_eo.size());
            for (i = 0; i < _eo.size(); i++)
                VarStp[i] = _eo.stdevs[i] * rng.normal();
            unsigned nq = _eo.correlations.size() - 1;
            for (k = 0; k < _eo.size()-1; k++)
            {
                n1 = _eo.size() - k - 1;
                n2 = _eo.size() - 1;
                for (i = 0; i < k; i++)
                {
                    d1 = VarStp[n1];
                    d2 = VarStp[n2];
                    S  = sin( _eo.correlations[nq] );
                    C  = cos( _eo.correlations[nq] );
                    VarStp[n2] = d1 * S + d2 * C;
                    VarStp[n1] = d1 * C - d2 * S;
                    n2--;
                    nq--;
                }
            }
            for (i = 0; i < _eo.size(); i++)
                _eo[i] += VarStp[i];
            bounds.foldsInBounds(_eo);
            return true;
        }


  private :

    /** Initialization of simple ES */
    void init(eoEsSimple<FitT>, eoEsMutationInit& _init)
    {
        unsigned size = bounds.size();
        TauLcl = _init.TauLcl();
        TauLcl /= sqrt(2*(double) size);
        std::cout << "Init<eoEsSimple>: tau local " << TauLcl << std::endl;
    }


    /** Initialization of standard ES

    @overload
    */
    void init(eoEsStdev<FitT>, eoEsMutationInit& _init)
    {
        unsigned size = bounds.size();
        TauLcl = _init.TauLcl();
        TauGlb = _init.TauGlb();
        // renormalization
        TauLcl /= sqrt( 2.0 * sqrt(double(size)) );
        TauGlb /= sqrt( 2.0 * double(size) );
        std::cout << "Init<eoStDev>: tau local " << TauLcl << " et global " << TauGlb << std::endl;
    }


    /** Initialization of full ES

    @overload
    */
    void init(eoEsFull<FitT>, eoEsMutationInit& _init)
    {
        init(eoEsStdev<FitT>(), _init);
        TauBeta = _init.TauBeta();
        std::cout << "Init<eoEsFull>: tau local " << TauLcl << " et global " << TauGlb << std::endl;
    }


    /** Local factor for mutation of std deviations */
    double TauLcl;

    /** Global factor for mutation of std deviations */
    double TauGlb;

    /** Factor for mutation of correlation parameters */
    double TauBeta;

    /** Bounds of parameters */
    eoRealVectorBounds& bounds;

    /** Minimum stdev.

    If you let the step-size go to 0, self-adaptation stops, therefore we give a
    lower bound. The actual value used is somewhat arbitrary and the is no
    theoretical reasoning known for it (Sep 2005).

    The code that we have in EO is a port from a C code that Thomas Bäck kindly
    donated to the community some years ago. It has been modified by Marc
    Schoenauer for inclusion in EvolC, than by Maarten Keijzer into EO. The
    exact value was adjusted based on practice.

    Removing this doesn't work well, but it was never tried to figure out what
    the best value would be.
    */
    static const double stdev_eps;
};


// Minimum value of stdevs, see declaration for details.
template <class EOT>
const double eoEsMutate<EOT>::stdev_eps = 1.0e-40;

#endif
