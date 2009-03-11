/*
 * Copyright (C) DOLPHIN Project-Team, Lille Nord-Europe, 2007-2008
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

#ifndef EODVRPSECONDSELAPSEDCONTINUE_H_
#define EODVRPSECONDSELAPSEDCONTINUE_H_

#include <eo>
 #include <eoContinue.h>

template< class EOT>



class eoDVRPSecondsElapsedContinue :  public eoContinue<EOT>
{
	time_t start;

	int seconds;

public:

	eoDVRPSecondsElapsedContinue(int nSeconds) :  start(time(0)), seconds(nSeconds) {}

   void resetStart(){ start = time(0);}


   virtual bool operator() ( const eoPop<EOT>& _pop ) {

	     time_t now = time(0);

         time_t diff = now - start;

         if (diff >= seconds)

        	 return false; //stop


         return true;

     }


   virtual std::string className(void) const { return "eoSecondsElapsedContinue"; }

     void readFrom (std :: istream & __is) {

       __is >> start >> seconds;
     }

     void printOn (std :: ostream & __os) const {

       __os << start << ' ' << seconds << std :: endl;
     }


};












#endif /*EODVRPSECONDSELAPSEDCONTINUE_H_*/
