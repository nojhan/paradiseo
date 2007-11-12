// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTwoOptMutation.h
// (c) GeNeura Team, 2000 - EEAAX 2000 - Maarten Keijzer 2000
// (c) INRIA Futurs - Dolphin Team - Thomas Legrand 2007
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
    		 thomas.legrand@lifl.fr
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef eoTwoOptMutation_h
#define eoTwoOptMutation_h

//-----------------------------------------------------------------------------


/**
 * Especially designed for combinatorial problem such as the TSP.
 */
template<class EOT> class eoTwoOptMutation: public eoMonOp<EOT>
{
 public:
 
  typedef typename EOT::AtomType GeneType;
 
  /// CTor
  eoTwoOptMutation(){}
  
  
  /// The class name.
  virtual std::string className() const { return "eoTwoOptMutation"; }


  /**
   * 
   * @param eo The cromosome which is going to be changed.
   */
  bool operator()(EOT& _eo)
    {    	
      unsigned i,j,from,to;
       
      // generate two different indices
      i=eo::rng.random(_eo.size());
      do j = eo::rng.random(_eo.size()); while (i == j);
     	
      from=std::min(i,j);
      to=std::max(i,j); 
	  int idx =(to-from)/2;
	  
	  // inverse between from and to
      for(unsigned int k = 0; k <= idx ;k++)
      	std::swap(_eo[from+k],_eo[to-k]);

      return true;
    }
    
};


//-----------------------------------------------------------------------------
#endif

