// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSGA.h
// (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000
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

#ifndef _eoSGA_h
#define _eoSGA_h

#include <eoOp.h>
#include <eoContinue.h>
#include <eoPop.h>
#include <eoSelectOne.h>
#include <eoSelectPerc.h>
#include <eoEvalFunc.h>
#include <eoAlgo.h>

template <class EOT>
class eoSGA : public eoAlgo<EOT>
{
public :
    eoSGA(
        eoContinue<EOT>& _cont, 
        eoMonOp<EOT>& _mutate, float _mrate,
        eoQuadraticOp<EOT>& _cross, float _crate,
        eoSelectOne<EOT>& _select,
        eoEvalFunc<EOT>& _eval) 
        : cont(_cont), 
          mutate(_mutate), 
          mutationRate(_mrate),
          cross(_cross),
          crossoverRate(_crate),
          select(_select),
          eval(_eval) {}

    void operator()(eoPop<EOT>& _pop)
    {
        eoPop<EOT> offspring;
        
        do
        {
            select(_pop, offspring);

            unsigned i;
	        
            for (i=0; i<_pop.size()/2; i++) 
            {
	          if ( rng.flip(crossoverRate) ) 
              {
		        // this crossover generates 2 offspring from two parents
		        cross(offspring[2*i], offspring[2*i+1]);
	          }
	        }

	        for (i=0; i < _pop.size(); i++) 
            {
	          if (rng.flip(mutationRate) ) 
              {
		        mutate(offspring[i]);
              }
	          
            }

            _pop.swap(offspring);
            apply<EOT>(eval, _pop);

        } while (cont(_pop));
    }

private :

    eoContinue<EOT>& cont;
    eoMonOp<EOT>& mutate;
    float mutationRate;
    eoQuadraticOp<EOT>& cross;
    float crossoverRate;
    eoSelectPerc<EOT> select;
    eoEvalFunc<EOT>& eval;
};

#endif
