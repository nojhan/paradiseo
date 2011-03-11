/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef EOCOUNTEDFILEMONITOR_H_
#define EOCOUNTEDFILEMONITOR_H_

#include <utils/eoFileMonitor.h>

class eoCountedFileMonitor: public eoFileMonitor
{
	public:
		eoCountedFileMonitor(unsigned int _frequency=1, std::string _filename="generation_data", std::string _delim="\t", bool _firstcall=false, bool _lastcall=true):
			eoFileMonitor(_filename, _delim, false, true), counter(0), frequency(_frequency), delim(_delim), firstcall(_firstcall), lastcall(_lastcall) {}

   	virtual std::string className(void) const { return "eoCountedFileMonitor"; }		
	eoMonitor& operator()(std::ostream& os)
	{
     	if( ! (++counter % frequency) || firstcall )
		{
			firstcall = false;
			os << counter << delim.c_str();
			iterator it = vec.begin();
		
			os << (*it)->getValue();
		
			for(++it; it != vec.end(); ++it)
			{
				os << delim.c_str() << (*it)->getValue();
			}
			os << '\n';
		}
		//counter++;
     	return *this;
	}

	void lastCall() { 
		if(lastcall && (counter-- % frequency))
		{
			firstcall = true;
			ofstream os(getFileName().c_str(), ios_base::app);
			this->operator()(os);
		}
	}

	private:
		unsigned int counter, frequency;
		std::string delim;
		bool lastcall, firstcall;
};					
#endif
