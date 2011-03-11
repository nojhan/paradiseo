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

#ifndef EOSINGLEFILECOUNTEDSTATESAVER_H_
#define EOSINGLEFILECOUNTEDSTATESAVER_H_
#include <utils/eoState.h>
#include <utils/eoUpdater.h>

class eoSingleFileCountedStateSaver : public eoUpdater
{
 public :
     eoSingleFileCountedStateSaver(unsigned _interval, const eoState& _state, std::string _filename, bool _saveOnFirstCall=false, bool  _saveOnLastCall=true )
       : interval(_interval), state(_state), filename(_filename),  saveOnFirstCall(_saveOnFirstCall), saveOnLastCall(_saveOnLastCall), counter(0)
		{
			os.open(filename.c_str());
			if (!os){
				std::string str = "eoFileMonitor: Could not open " + filename;
				throw std::runtime_error(str);			
			}
		}
 
   	virtual std::string className(void) const { return "eoSingleFileCountedStateSaver"; }

	
	void operator()(void)
	{
		if ( !(++counter % interval) || saveOnFirstCall)
		{
			doItNow();
			saveOnFirstCall = false;
		}
	}
	
	void lastCall(void)
	{
		if (saveOnLastCall && (counter-- % interval))
		{
			saveOnFirstCall = true;
			this->operator()();
		}
	}


	private:

     const eoState& state;
     unsigned interval;
     unsigned counter;
     bool saveOnLastCall;
	 bool saveOnFirstCall;
	 std::ofstream os;
	 std::string filename;


	void doItNow(void)
	{
		os << "Generation Number :" << counter << std::endl;
		state.save(os);
	}

 };
#endif
