// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
/*
-----------------------------------------------------------------------------
File............: t-eo2dVector.cc
Author..........: Geneura Team (this file: Victor Rivas, vrivas@ujaen.es)
Date............: 01-Oct-1999, at Fac. of Sciences, Univ. of Granada (Spain)
Description.....: Test for 2 dimensional eoVector.

  ================  Modif. 1  ================
  Author........:
  Date..........:
  Description...:

-----------------------------------------------------------------------------
*/

#include <stdexcept>
#include <paradiseo/eo/eo2dVector.h>  // eo2dVector
#include <paradiseo/eo/eoUniform.h>  // Random generator

//-----------------------------------------------------------------------------

typedef unsigned T;
typedef double fitnessT;
typedef eo2dVector<T,fitnessT> C;

//-----------------------------------------------------------------------------

main()
{
  {
    C c1;
    cout << "Default constructor: " << endl
	 << c1 << endl;
  }
  {
    C c1( 5,6,1 );
    cout << "Default constructor with values: " << endl
	 << c1 << endl;

  }
  {
    eoUniform<T> aleat( 1,10 );
    C c1( 5,6, aleat );
    cout << "Random constructor: " << endl
	 << c1 << endl;

  }
  {
    C c1( 3,4,1 ), c2( c1 );
    cout << "Copy constructor: " << endl
	 << "Original chromosome: " << endl
	 << c1 << endl
	 << "Copy chromosome: " << endl
	 << c2 << endl;
  }

  eoUniform<T> aleat( 1,10 );
  C c1( 3,4,aleat );

  cout << "-----------------------------------------------------" << endl
       << "Since now on all the operations are applied to " << endl
       << c1
       << "-----------------------------------------------------" << endl;
  {
    cout << "getGene(2,2): "
	 << c1.getGene(2,2) << endl;
  }
  {
    c1.setGene( 2,2,300 );
    cout << "setGene(2,2,300): " << endl
	 << c1 << endl;
  }
  {
    unsigned u1=0, u3=333, u5=555;
    vector<T> v1( 4,u1 ), v2( 4,u3 ), v3( 4,u5 );
    c1.insertRow( 0,v1 );
    c1.insertRow( 3,v2 );
    c1.insertRow( 5,v3 );
    cout << "Insert rows at positions 0, 3 and 5: " << endl
	 << c1 << endl;
  }
  {
    c1.deleteRow( 5 );
    c1.deleteRow( 3 );
    c1.deleteRow( 0 );
    cout << "Delete rows at positions 5, 3 and 0: " << endl
	 << c1 << endl;
  }
  {
    unsigned u1=0, u3=333, u6=666;
    vector<T> v1( 3,u1 ), v2( 3,u3 ), v3( 3,u6 );
    c1.insertCol( 0,v1 );
    c1.insertCol( 3,v2 );
    c1.insertCol( 6,v3 );
    cout << "Insert columns at positions 0, 3 and 6: " << endl
	 << c1 << endl;
  }
  {
    c1.deleteCol( 6 );
    c1.deleteCol( 3 );
    c1.deleteCol( 0 );
    cout << "Delete columns at positions 6, 3 and 0: " << endl
	 << c1 << endl;
  }
  {
    cout << "Number of Rows: " << endl
	 << c1.numOfRows() << endl;
  }
  {
    cout << "Number of Columns: " << endl
	 << c1.numOfCols() << endl;
  }

  {
    cout << "Class Name: " << endl
	 << c1.className() << endl;
  }


  cout << "-----------------------------------------------------" << endl
       << "Catching exceptions: " << endl
       << c1
       << "-----------------------------------------------------" << endl;
  {
    cout << "* Trying getGene(10,1): " << endl;
    try {
      c1.getGene( 10,1 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }
  {
    cout << "* Trying getGene(1,10): " << endl;
    try {
      c1.getGene( 1,10) ;
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }
  {
    cout << "* Trying setGene( 10,1,999 ): " << endl;
    try {
      c1.setGene( 10,1,999 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }
  {
    cout << "* Trying setGene( 1,10,999 ): " << endl;
    try {
      c1.setGene( 1,10,999 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }
  {
    unsigned u1=111;
    vector<T> v1( 4, u1 );
    cout << "* Trying insertRow( 10, v1 ): " << endl;
    try {
      c1.insertRow( 10,v1 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }
  {
    unsigned u1=111;
    vector<T> v1( 5, u1 );
    cout << "* Trying insertRow( 1, v1 ) with v1.size()=5: " << endl;
    try {
      c1.insertRow( 1,v1 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }

  {
    cout << "* Trying deleteRow( 10 ): " << endl;
    try {
      c1.deleteRow( 10 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }

  {
    unsigned u1=111;
    vector<T> v1( 3, u1 );
    cout << "* Trying insertCol( 10,v1 ): " << endl;
    try {
      c1.insertCol( 10,v1 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }

  {
    unsigned u1=111;
    vector<T> v1( 5, u1 );
    cout << "* Trying insertCol( 1,v1 ) with v1.size()=5: " << endl;
    try {
      c1.insertCol( 1,v1 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }

  {
    cout << "* Trying deleteCol( 10 ): " << endl;
    try {
      c1.deleteCol( 10 );
    } catch (exception& e ) {
      cerr << e.what() << endl;
    }
  }

}

//-----------------------------------------------------------------------------
