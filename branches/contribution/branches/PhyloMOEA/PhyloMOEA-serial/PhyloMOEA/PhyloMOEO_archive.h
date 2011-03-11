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
#ifndef PHYLOMOEO_ARCHIVE_H
#define PHYLOMOEO_ARCHIVE_H

#include <PhyloMOEO.h>

typedef  moeoArchive<PhyloMOEO> PhyloMOEOArchive;
typedef  PhyloMOEOArchive PhyloMOEOPFArchive;




class PhyloMOEOParetoSolutionsArchive:public moeoUnboundedArchive<PhyloMOEO>
{
	public:
		void save_trees( std::string filename, string title="" )
		{
			create_file( filename );
			if(title.size()>0) os << title << endl;
			os << size() << std::endl;
			for(int i=0; i<size(); i++)	
				operator[](i).get_tree().printNewick(os);
			os.close();
		}

		void save_scores( std::string filename, string title = "" )
		{
			create_file( filename );
			if(title.size()>0) os << title << endl;
			for(int i=0; i<size(); i++)	
				os << operator[](i) << std::endl;
			os.close();
		}
	protected:
		std::ofstream os;
	private:
		void create_file( std::string filename)
		{
			os.open(filename.c_str());
			if (!os){
				std::string str = "Could not open " + filename;
				throw std::runtime_error(str);			
			}
		}
};

class PhyloMOEODummyArchive:public PhyloMOEOParetoSolutionsArchive
{
	public:

	void update(const eoPop < PhyloMOEO > & _pop)
	{
		std::copy(_pop.begin(), _pop.end(), back_inserter(*this) );
	}
};

class PhyloMOEOFinalSolutionsArchive: public PhyloMOEOParetoSolutionsArchive
{
  public:
	bool operator()(const eoPop < PhyloMOEO > & _pop) { return update(_pop); }
  protected:

	bool update(const eoPop < PhyloMOEO > & _pop)
	{
		
		std::copy(_pop.begin(), _pop.end(), back_inserter(*this) );

		moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;	
		for(int i=0; i< size()-1; i++)
		{
			//cout << (i+1)/(p.currentSize()*1.0) << "%" << endl;
			for(int j=i+1; j< size(); j++)
			{
				phylotreeIND &sol = operator[](i).get_tree();
				phylotreeIND &sol_2 = operator[](j).get_tree();
				if( sol.compare_topology_2( sol_2) == 0)
				{
					if( paretoComparator( operator[](i).objectiveVector(), operator[](j).objectiveVector() ) )
					{
						operator[](i) = back();
						pop_back();
						i--; j = size();
					}
					else
					{
						operator[](j) = back();
						pop_back();
						j--;
					}
				}
			}
		}
		return true;
   }
};
#endif
