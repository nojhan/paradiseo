/*
   RoyalRoad.h
   -- Implementation of the Royal Road function for any length and block size
   (c) GeNeura Team 2001, Marc Schoenauer 2000

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
CVS Info: $Date: 2001-06-21 12:03:17 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/test/RoyalRoad.h,v 1.3 2001-06-21 12:03:17 jmerelo Exp $ $Author: jmerelo $
*/

#ifndef RoyalRoad_h
#define RoyalRoad_h

template<class EOT>
class RoyalRoad: public eoEvalFunc<EOT> {

 public:

  typedef typename EOT::Fitness FitT;

  /// Ctor: takes a length, and divides that length in equal parts
  RoyalRoad( unsigned _div ): eoEvalFunc<EOT >(), div( _div ) {};

  // Applies the function
  virtual void operator() ( EOT & _eo )  {
	FitT fitness = 0;
    if (_eo.invalid()) {
	  for ( unsigned i = 0; i < _eo.size()/div; i ++ ) {
		bool block = true;
		for ( unsigned j = 0; j < div; j ++ ) {
		  block &= _eo[i*div+j];
		}
		if (block) {
		  fitness += div;
		}
	  }
	  _eo.fitness( fitness );
	}
  };

  private:
	unsigned div;

};

#endif
