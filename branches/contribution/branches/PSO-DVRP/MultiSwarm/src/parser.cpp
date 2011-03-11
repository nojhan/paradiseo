/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord-Europe, 2007-2008
 * (C) OPAC Team, LIFL, 2002-2008
 *
 * (c) Mostepha Redouane Khouadjia <mr.khouadjia@ed.univ-lille1.fr>, 2008
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

#include"parser.h"



void IsReadable(ifstream& _filein)
	{

		if ( !_filein)
	    {
	        cerr << "Opening file is fail\n";
            //exit(1);
	    }


	}

string ReadableLine(ifstream& _filein)
	{
		string line;

		if(!(getline(_filein, line )))
		{
			cerr << "Error was occur when read line\n";
			exit(1);
		}
		return line;
	}


	void  IsCreatable(ofstream& _fileout)
	{
		if ( !_fileout )
		   {
			  cerr << "Error occur when file was created\n";
			  exit(1);
	       }
		   cout<<" File was created"<<endl;
	}



	/*
	string fileOut(argv[1]);
	const string BENCHMARK  ="/home/mustapha/framework/paradiseo-1.1/paradiseo-dynamic-schedule/application/Benchmarks/"+fileOut+".vrp";
	std::istringstream iss(argv[2]); iss >> INERTIA; iss.clear();
	iss.str(argv[3]); iss >> LEARNING_FACTOR1;  iss.clear();
	iss.str(argv[4]); iss >> LEARNING_FACTOR2;  iss.clear();
	iss.str(argv[5]); iss >> SEED; rng.reseed(SEED); iss.clear();

	string  extension = "";
	extension = extension +"-IN(" + argv[2] + ")" + "-C1(" + argv[3] + ")" + "-C2("+ argv[4] + ")" ;

*/

