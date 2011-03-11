/*
 * Copyright (C) DOLPHIN Project-Team, INRIA Lille Nord-Europe, 2007-2008
 * (C) OPAC Team, LIFL, 2002-2008
 *
 redouanedz
 àà   * (c) Mostepha Redouane Khouadjia <mr.khouadjia@ed.univ-lille1.fr>, 2008
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

#include "eoEventScheduler.h"
#include "eoGlobal.h"


eoEventScheduler::eoEventScheduler(unsigned _nbrts,double  _tsim, double _tadvance,

								 double _tcutoff, const string _filename ):

    nbrts(_nbrts),tsim(_tsim), tadvance(_tadvance), tcutoff(_tcutoff), tstep(0.0), tday(0.0), tslice(0.0), filename(_filename){

	FindTimeDay();

	SetFleetCapacity();

	SetTimeSlice();

	TIME_STEP =0.0;

	TIME_ADVANCE = _tadvance * tday;

   }

 eoEventScheduler::~eoEventScheduler(){}

  double  eoEventScheduler::TimeOfSimulation(){return tsim;}


	double eoEventScheduler::TimeSlice(){return tslice;}

	double eoEventScheduler::TimeStep(){ return tstep;}

	double eoEventScheduler::TimeCutoff() {return tcutoff;}

	double eoEventScheduler::TimeAdvance(){return tadvance;}

	void eoEventScheduler::UpdateTimeStep()

	{tstep+= tslice;

	 TIME_STEP = tstep;}

	void eoEventScheduler::setTimeStep(double _tstep){tstep = TIME_STEP = _tstep;}

	void eoEventScheduler::setTimeAdvance(double _tadvance){ tadvance = TIME_ADVANCE =_tadvance ;}

	double  eoEventScheduler::TimeDay(){ return tday;}

	std::string eoEventScheduler::TimeToString (double _time){

		std::ostringstream oss;

		oss << _time;

		return oss.str();
	}


	void eoEventScheduler::SetTimeSlice() { TIME_SLICE = tslice = tday/nbrts;}


	void eoEventScheduler::SetTimeDay(const double  _tday){tday = TIME_DAY= _tday;}

	unsigned eoEventScheduler::CapacityTour() {return VEHICULE_CAPACITY;}

	void eoEventScheduler::FindTimeDay(){


		ifstream file (filename.c_str());

		string line;

		unsigned int customer;

		double time;

		istringstream  iss;

		do

			getline(file,line);


		while(line.find("DEPOT_TIME_WINDOW_SECTION")== string::npos);


		getline(file,line);

		iss.str(line);

		iss>>customer>>customer>>time;

		iss.clear();

		SetTimeDay(time);

		file.close();

	}

	void eoEventScheduler::SetFleetCapacity(){

		   istringstream  iss;

		   string line;

		   ifstream filein (filename.c_str());

		   IsReadable(filein);



			do

				getline(filein,line);

			while(line.find("NUM_VEHICLES")== string::npos);



			iss.str(line);

			iss>>line>>FLEET_VEHICLES;

			iss.clear();



			do
				getline(filein,line);

			while(line.find("CAPACITIES")== string::npos);


			iss.str(line);

			iss>>line>> VEHICULE_CAPACITY;

			iss.clear();


			filein.close();

	}


	void eoEventScheduler::GenerateBenchmark(){

		istringstream  iss;  //Input stream

		string fileoutname;

		unsigned int customer,TimeOrder,i;  // Set of Costumers to serve, and  available time of custumer order

		const string fileiname = filename;

		vector<unsigned int> customers;

		string line,

		str = fileiname;

		str = str.erase(str.size() - 4); // Ajust the name of generated file according to the time slice.


	  while(TimeStep()<= TimeDay())
		{

		  ifstream filein (fileiname.c_str());

		  IsReadable(filein);  //Read benchmark file


	      fileoutname = str+ "out." + TimeToString(tstep)+".txt";  // The generated file name is terminated by ".out.txt"


		  ofstream fileout(fileoutname.c_str());

		  cout<<">>>>>>>> TimeStep "<<TimeStep()<<endl;


		do{
			getline(filein,line);

		 	fileout<< line<<endl;

   		}while (line.find("DEMAND_SECTION")== string::npos);




		do
			getline(filein,line);

		while(line.find("TIME_AVAIL_SECTION")== string::npos);


		getline(filein,line);

		do{

		   iss.str(line);

		   iss>>customer>>TimeOrder;

		   iss.clear();



		   if(TimeStep() == 0)

			   {if (TimeOrder > TimeCutoff() * TimeDay())

				   customers.push_back(customer);

			   }
		   else


		   if(TimeStep() > TimeCutoff() * TimeDay())

				   {
			   			if ((TimeOrder > (TimeStep() - TimeSlice())) && (TimeOrder <= TimeCutoff() * TimeDay()))

					   customers.push_back(customer);

				   }
		   else

			   if ((TimeOrder > (TimeStep()- TimeSlice()) && TimeOrder <= TimeStep()))

				      customers.push_back(customer);




	        getline(filein,line);

		}while(line.find("EOF")== string::npos);

		for ( size_t k = 0; k < customers.size(); ++k)

			cout <<customers[k] << '\t';




		 filein.seekg (std::ios_base::beg); //Return to the beginning of the benchmark file for gather informations about custumers

		do

			getline(filein,line);

		while (line.find("DEMAND_SECTION")== string::npos);

				i=0;

				getline(filein,line);

				while( i< customers.size() && line.find("LOCATION_COORD_SECTION")== string::npos)
				{

					iss.str(line); //Copy string in the stream

					iss>>customer; // Move in the stream

					iss.clear();

					if(customer==customers[i])
					{
						fileout <<line<<endl;

						i++;
					}

					getline(filein,line);
				}

				while(line.find("LOCATION_COORD_SECTION")== string::npos)

				getline(filein,line);

				fileout<<line<<endl;

				getline(filein,line);

				i=0;

				while( i< customers.size() && line.find("DEPOT_LOCATION_SECTION")== string::npos)
						{
							iss.str(line);

							iss>>customer;

							iss.clear();

							if(customer==customers[i])
							{

								fileout <<line<<endl;

								i++;
							}

							getline(filein,line);
						}


				while(line.find("DURATION_SECTION")== string::npos)

					getline(filein,line);

				i=0;

				fileout<<line<<endl;

				getline(filein,line);

				while( i< customers.size() && line.find("DEPOT_TIME_WINDOW_SECTION")== string::npos)

				{
					iss.str(line);

					iss>>customer;

					iss.clear();

					if(customer==customers[i])

					{

						fileout <<line<<endl;

						i++;
					}

					getline(filein,line);

				}



				while(line.find("DEPOT_TIME_WINDOW_SECTION")== string::npos)

					getline(filein,line);


				fileout<<line<<endl;

				getline(filein,line);


			    fileout<<line<<endl;


			    while(line.find("TIME_AVAIL_SECTION")== string::npos)

			    	getline(filein,line);


			    i=0;

			    fileout<<line<<endl;

			    getline(filein,line);

			    while( i< customers.size() && line.find("EOF")== string::npos)
					{

			    	iss.str(line);

			    	iss>>customer;

			    	iss.clear();

			    	if(customer==customers[i])

			    	{

			    		fileout <<line<<endl;

			    		i++;

			    	}

			    	getline(filein,line);

					}

			    fileout<<"EOF"<<endl;   //Terminate new benchmark file

			    customers.clear();

			    cout<<endl<<"The Benchmarks was generated in the step of time :"<< TimeStep()<<endl;

			    filein.close();

			    fileout.close();


			    if (TimeStep() > TimeCutoff()*TimeDay())

			    	break;
			    else

			    UpdateTimeStep();

					    }

	  setTimeStep(0);

	  cout <<endl<<"-------------All Benchmarks were successfully generated-------------"<<endl;

	}




