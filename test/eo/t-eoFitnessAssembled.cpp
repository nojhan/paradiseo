// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// t-eoFitnessAssembled.cpp
// Marc Wintermantel & Oliver Koenig
// IMES-ST@ETHZ.CH
// March 2003

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
	     Marc.Schoenauer@inria.fr
	     mak@dhi.dk
*/
//-----------------------------------------------------------------------------
#include <iostream>
#include <stdexcept>

#include <paradiseo/eo.h>
//#include "eoScalarFitnessAssembled.h"

void test_eoScalarFitnessAssembledClass(){

  // Create instances
  eoAssembledMinimizingFitness A,B,C(5, 1.3, "C value");

  // Add some values to them
  A.push_back( 5.6, "first value"  );
  A.push_back( 3.2, "second value" );
  A.push_back( 2.6, "third value" );

  B.push_back( 1.2 );
  B.push_back( 3.2 );
  B.push_back( 5.2 );

  B.setDescription( 1, "B descr" );

  std::cout << "Created instances A,B and C, added some vals; testing << operator " << std::endl;
  std::cout << "A= " << A << std::endl;
  std::cout << "B= " << B << std::endl;
  std::cout << "C= " << C << std::endl;
  std::cout << "Printing values and descriptions: " << std::endl;
  std::cout << "A: "; A.printAll( std::cout ); std::cout << std::endl;
  std::cout << "B: "; B.printAll( std::cout ); std::cout << std::endl;
  std::cout << "C: "; C.printAll( std::cout ); std::cout << std::endl;

  A.resize(8, 100.3, "A resized");
  std::cout << "Resized A: "; A.printAll( std::cout ); std::cout << std::endl;

  std::cout << "Access fitness values of A and B: " << "f(A)= " << (double) A << " f(B)= " << (double) B << std::endl;

  // Testing constructors and assignments
  eoAssembledMinimizingFitness D(A) ,E(3.2);
  std::cout << "D(A) = " << D << "\t" << "E(3.2)= " << E << std::endl;
  eoAssembledMinimizingFitness F,G;
  F=A;
  G= 7.5;
  std::cout << "F = A : " << F << "\t G = 7.5 : " << G << std::endl;

  // Comparing...
  std::cout << "A<B: " << (A<B) << std::endl;
  std::cout << "A>B: " << (A>B) << std::endl;
  std::cout << "A<=B: " << (A<=B) << std::endl;
  std::cout << "A>=B: " << (A>=B) << std::endl;

}



int main(){

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "START t-eoFitnessAssembled" << std::endl;

  try{
    // Test the fitness class itself
    test_eoScalarFitnessAssembledClass();



  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    return 1;
  }

  std::cout << "END  t-eoFitnessAssembled" << std::endl;
  std::cout << "----------------------------------" << std::endl;

  return 0;

}
