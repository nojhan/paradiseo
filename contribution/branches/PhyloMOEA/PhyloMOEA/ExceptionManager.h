/***************************************************************************
 *   Copyright (C) 2005 by Waldo Cancino                                   *
 *   wcancino@icmc.usp.br                                                  *
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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#ifndef _ExceptionManager_H_
#define _ExceptionManager_H_

using namespace std;

class ExceptionManager
{
	string msg;
	short number;
	public:
	ExceptionManager(int n) { 
		number = n; 
		switch(number){
			case 10:
				msg = "Could not open file\n";
				break;
			case 11:
				msg ="Incorrect file format\n";
				break;
			case 12:
				msg = "Could not write to file\n";
				break;
		}	
	};
	void Report()
	{
		cout << endl << msg << endl;
		cout << "PhyloMOEA was finished because errors." << endl;
		exit(1);
	}
};

#endif

