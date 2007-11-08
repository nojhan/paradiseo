/* <peoPSOVelocity.h>
*
*  (c) OPAC Team, October 2007
*
* Clive Canape
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
*   Contact: clive.canape@inria.fr
*/

#ifndef _peoPSOVelocity_h
#define _peoPSOVelocity_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <utils/eoRNG.h>
#include <eoFunctor.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <eoReplacement.h>
#include <utils/eoHowMany.h>

template <class POT>
class peoPSOVelocity : public eoReplacement<POT>
{
    public:
    
    	typedef typename POT::ParticleVelocityType VelocityType;
    
        peoPSOVelocity(	const double & _c3,
        				eoVelocity < POT > &_velocity):
        				c3 (_c3),
        				velocity (_velocity){}

        void operator()(eoPop<POT>& _dest, eoPop<POT>& _source)
        {
        	
        	VelocityType newVelocity,r3;
        	r3 =  (VelocityType) rng.uniform (1) * c3;
        	for(unsigned i=0;i<_dest.size();i++)
        		for(unsigned j=0;j<_dest[i].size();j++)
        		{
        			newVelocity=  _dest[i].velocities[j] + r3 * (_source[0].bestPositions[j] - _dest[i][j]);
            		_dest[i].velocities[j]=newVelocity;
        		}  
        		     
        }
        
    protected:
    	const double & c3;
    	eoVelocity < POT > & velocity;
};
#endif

