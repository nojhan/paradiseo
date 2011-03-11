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
#ifndef PHYLOMOEOPARTITONSTAT_H_
#define PHYLOMOEOPARTITONSTAT_H_

#include <moeoObjVecStat.h>
#include <PhyloMOEO.h>
//#include <utils.h>

struct part_info
{
	int freq; double prob;
	double dp, dl, tdp, tdl;
	short int type; // 001=intermediate 1, 010=mp 2, 100=ml 4, 
					// 110= mp & ml 6, 101 = ml & int 5 011 mp & int 3 , 111 tree 7
	bool inverse;
	part_info(): freq(0), prob(0.0), dp(0), dl(0), tdp(0), tdl(0), type(0), inverse(false) {};
};

typedef std::map<string,struct part_info> partition_map;


 class PhyloMOEOPartitionStat :  public eoStat<PhyloMOEO, partition_map>
 {
 public :
 
     using eoStat<PhyloMOEO, partition_map>::value;
 
     PhyloMOEOPartitionStat(std::string _description = "Partition Stats")
       : eoStat<PhyloMOEO, partition_map>(partition_map(), _description) {}
 
     virtual void operator()(const eoPop<PhyloMOEO>& _pop){
       doit(_pop); // specializations for scalar and std::vector
     }
 
   virtual std::string className(void) const { return "PhyloMOEOPartitionStat"; }

 
 private :
 
     // Specialization for pareto fitness
     void doit(const eoPop<PhyloMOEO>& _pop)
     {
		value().clear();
		calculate_frequence_splits( _pop, value() ); 
     }
 
	void calculate_frequence_splits( const eoPop<PhyloMOEO> &pop, partition_map &split_frequences)
	{
		bool ismp, isml;
		isml = isml = false;
		double dtp, dtl, maxdl, maxdp, dp, dl;
		int n = pop.size();
		double best_ml_score, best_mp_score; 
		string split, split_i;
		graph::edge_iterator it, it_e;
		//print_media_scores(pop, -1, best_l_idx, best_p_idx);
		moeoBestObjVecStat<PhyloMOEO> best_inds;
		best_inds(pop);
		best_mp_score = (best_inds.value())[0];
		best_ml_score = (best_inds.value())[1];
		const PhyloMOEO &best_l = best_inds.bestindividuals(1);
		const PhyloMOEO &best_p = best_inds.bestindividuals(0);
		maxdl = maxdp = 0;
		for(int i=0; i<n; i++)
		{
			dl = dtp = dtl = dp = 0;
			isml = ismp = false;
			const PhyloMOEO &sol = pop[i];
			if(!sol.get_tree().splits_valid())sol.get_tree().calculate_splits_exp();
			it = sol.get_tree().TREE.edges_begin();
			it_e = sol.get_tree().TREE.edges_end();
			// is the mp or ml tree
			if( sol.objectiveVector().operator[](1) == best_ml_score) isml=true;
			if( sol.objectiveVector().operator[](0) == best_mp_score) ismp= true;
			dtl = sol.get_tree().compare_topology_2( best_l.get_tree())	/ (2.0*(sol.get_tree().TREE.number_of_edges() - sol.get_tree().number_of_taxons()));
			//sol->compare_topology_3( *best_l);
			dtp = sol.get_tree().compare_topology_2( best_p.get_tree())	/ (2.0*(sol.get_tree().TREE.number_of_edges() - sol.get_tree().number_of_taxons()));
			//sol->compare_topology_3( *best_p);
			while(it!=it_e)
			{
				if( sol.get_tree().is_internal( *it))
				{
					split = sol.get_tree().get_split_key( *it);
					split_i = sol.get_tree().get_invert_split_key( *it);
					if( split_frequences.find(split) != split_frequences.end() )
					{
						split_frequences[split].freq++;	
						split_frequences[split].prob += 1.0/n;	
					}
					else if( split_frequences.find(split_i) != split_frequences.end() )
					{
						split_frequences[split_i].freq++;
						split_frequences[split_i].prob += 1.0/n;	
						split = split_i;
					}
					else
					{
						split_frequences[split].freq = 1;
						split_frequences[split].prob = 1.0/n;	
					}
					if( isml )
						split_frequences[split].type |= 4; // split belongs ml tree
					if( ismp )
						split_frequences[split].type |= 2; // split belongs mp tree
					if( !(isml || ismp)) 	split_frequences[split].type |= 1; // split belong intermediate
					dp = sol.objectiveVector().operator[](0) - best_mp_score;
					dl = sol.objectiveVector().operator[](1) - best_ml_score;
					split_frequences[split].dp += dp;
					split_frequences[split].dl += dl;
					maxdp = (dp > maxdp ? dp : maxdp);
					maxdl = (dl > maxdl ? dl : maxdl);
					split_frequences[split].tdp += dtp;
					split_frequences[split].tdl += dtl;
				}
				++it;
			}
			//sol->invalidate_splits();
			sol.get_tree().remove_split_memory();
		}
	
		partition_map::iterator it1 = split_frequences.begin();
		partition_map::iterator it2 = split_frequences.end();
		
		while(it1!=it2)
		{
			if(maxdp!=0) (*it1).second.dp /= (1.0*(*it1).second.freq*maxdp);
			if(maxdl!=0)(*it1).second.dl /= (1.0*(*it1).second.freq*maxdl);
			(*it1).second.tdp /= (1.0*(*it1).second.freq);
			(*it1).second.tdl /= (1.0*(*it1).second.freq); 
			++it1;
		}
	}
	
 
 };

std::string inverse_split( string split )
	{
		string s;
		string::iterator it1 = split.begin();
		string::iterator it2 = split.end();
		while(it1!= it2)
		{
			s += *it1 == '*' ? '.' : '*';
			++it1;
		}
		return s;
	}


std::ostream & operator<<(std::ostream & _os, const partition_map & _map) { 
	partition_map::const_iterator it1 = _map.begin();
	partition_map::const_iterator it2 =  _map.end();
	_os << _map.size() << endl;
	_os.setf(ios::fixed);
	while(it1!=it2)
	{
			_os << 
				((*it1).second.inverse ? inverse_split((*it1).first) : (*it1).first)
				<< '\t' << (*it1).second.freq << '\t' << (*it1).second.prob << '\t'
				<< (*it1).second.dp << '\t' << (*it1).second.dl << '\t'
				<< (*it1).second.tdp << '\t' << (*it1).second.tdl << '\t' << (*it1).second.type << endl;
			++it1;
	}
	return _os;
}

std::istream & operator>>(std::istream & _is, const partition_map & _map) { std::cout << "not implemented\n"; return _is; }
#endif
