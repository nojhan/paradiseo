// j'ai installé le svn :)
// re-test

/*
 * <moeo2DMinHypervolumeArchive.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * Arnaud Liefooghe
 * Jérémie Humeau
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
//-----------------------------------------------------------------------------

#ifndef MOEO2DMINHYPERVOLUMEARCHIVE_H_
#define MOEO2DMINHYPERVOLUMEARCHIVE_H_

#include <set>
#include <climits>


template < class MOEOT >
struct comp
{
	// returns a "before" b
	// all objectives = min
	bool operator() (const MOEOT & a, const MOEOT & b)
	{
		return  ((a.objectiveVector()[1] < b.objectiveVector()[1]) || ((a.objectiveVector()[1] == b.objectiveVector()[1]) && (a.objectiveVector()[0] < b.objectiveVector()[0])));
	}
};


/** 2D (minimization) bounded archive by hypervolume , base on a set */
template < class MOEOT >
class moeo2DMinHypervolumeArchive : public std::set<MOEOT , comp < MOEOT > >
{
public:

	typedef typename MOEOT::Fitness Fitness;
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	typedef typename std::set < MOEOT, comp<MOEOT>  >::iterator Iterator;

	using std::set < MOEOT, comp<MOEOT> > :: begin;
	using std::set < MOEOT, comp<MOEOT> > :: end;
	using std::set < MOEOT, comp<MOEOT> > :: insert;
	using std::set < MOEOT, comp<MOEOT> > :: erase;
	using std::set < MOEOT, comp<MOEOT> > :: size;
	using std::set < MOEOT, comp<MOEOT> > :: upper_bound;


	/**
	 * Ctr.
	 * @param _maxSize size of the archive (must be >= 2)
	 * @param _maxValue fitness assigned to the first and the last solution in the archive (default LONG_MAX)
	 */
	moeo2DMinHypervolumeArchive(unsigned int _maxSize=100, double _maxValue=LONG_MAX) : std::set < MOEOT, comp<MOEOT> > (), maxSize(_maxSize), maxValue(_maxValue)
	{
		maxSize = std::max((unsigned int) 2, maxSize);
	}


	/**
	 * Update the archive with a solution
	 * @param _moeo a solution
	 * @return true if _moeo has been added to the archive
	 */
	bool operator()(const MOEOT & _moeo)
	{
		//store result
		bool result;
		Iterator it;

		//If archive is empty -> add the sol and affect its fitness value
		if (size()==0)
		{
			result = true;
			insert(_moeo);
			it=begin();
			fitness(it, maxValue);
		}
		else   // test if sol can be added to the archive
		{
			result = insert(_moeo.objectiveVector());
			if (result)
			{
				if(size() < maxSize)
				{
					// if yes, insert it and recompute fitness value of MOEOT and its neighbors
					insert(hint, _moeo);
					if(size() > 2)
					{
						//general case
						hint--;
						computeFitness(hint);
					}
					else
					{
						//archive size <= 2, fitness=maxValue for each sol
						it=begin();
						while(it!=end())
						{
							fitness(it, maxValue);
							it++;
						}
					}
				}
				else
				{
					result = filter(_moeo);
				}
			}
		}
		return result;
	}


	/**
	 * update the archive with a population
	 * @param _pop a pop
	 * @return true if at least one solution of _pop has been added to the archive
	 */
	bool operator()(const eoPop < MOEOT > & _pop)
	{
		bool result = false;
    	bool tmp = false;
        for (unsigned int i=0; i<_pop.size(); i++)
        {
        	std::cout << "insert " << _pop[i].objectiveVector()[0] << ", " << _pop[i].objectiveVector()[1] << std::endl;
            tmp = (*this)(_pop[i]);
            result = tmp || result;
        }
        return result;
	}


	/**
	 * Test if insertion wrt Pareto-dominance is possible, and fix 'hint' if possible
	 * @param _objVec the objective vector of the sol to insert
	 * @return true if objVec can be added to the archive wrt Pareto-dominance
	 */
	bool insert(const ObjectiveVector & _objVec)
	{
		bool result = false;
		Iterator it;
		double min;
		// set the objVec to the empty solution
		empty.objectiveVector(_objVec);
		// compute the position where it would possibly be added
		it = upper_bound(empty);
		// compute the weigth from the previous solution
		min = begin()->objectiveVector()[0];
		if (it != begin())
		{
			it--;
			min = (*it).objectiveVector()[0];
			it++;
		}
		// if it has a better weitgh, or if it's an extreme sol, let's add it
		if (it == begin() || _objVec[0]<min)
		{
			// remove dominated solutions
			remove(it,_objVec);
			// set hint to the current iterator (probably modified by "remove")
			hint=it;
			// set result to true
			result = true;
		}
		return result;
	}


	/**
	 * print objective vector and fitness value of the archive
	 */
	void print()
	{
		Iterator it = begin();
		while(it!=end())
		{
			std::cout << (*it).objectiveVector()[0] << " " << (*it).objectiveVector()[1] << ", fit: " << (*it).fitness() << std::endl;
			it++;
		}
	}


protected:

	/** size max of the archive */
	unsigned int maxSize;
	/** fitness assigned to the first and the last solution in the archive */
	double maxValue;
	/** hint for the insertion */
	Iterator hint;

	/** an empty MOEOT used for checking insertion */
	MOEOT empty;


	/**
	 * set fitness
	 */
	void fitness(Iterator & _it, double _fitnessValue)
	{
		MOEOT* tmp;
		tmp = (MOEOT*)&(*_it);
		tmp->fitness(_fitnessValue);
	}


	/**
	 * remove solutions from the archive that are dominated by _objVec
	 * @param _it an iterator beginning on the first potentialy sol to remove
	 * @param _objVec the objective vector of the new solution
	 */
	void remove(Iterator & _it, const ObjectiveVector & _objVec)
	{
		Iterator itd;
		while ((_it!=end()) && ((*_it).objectiveVector()[0] >= _objVec[0]))
		{
			itd = _it;
			_it++;
			erase(itd);
		}
	}


	/**
	 * compute fitness value of a solution and its two neighbors
	 * @param _it refer to the solution
	 */
	void computeFitness(Iterator & _it)
	{
		Iterator tmp;
		if(_it!=begin())
		{
			tmp=_it;
			tmp--;
			compute(tmp);
		}
		_it++;
		if(_it!=end())
		{
			_it--;
			tmp=_it;
			tmp++;
			compute(tmp);
		}
		else
		{
			_it--;
		}
		compute(_it);
	}


	/**
	 * compute fitness value of a solution
	 * @param _it refer to the solution
	 */
	void compute(Iterator & _it)
	{
		double x0, x1, y0, y1, fit;
		if (_it==begin())
		{
			fitness(_it, maxValue);
		}
		else if ((++_it)==end())
		{
			_it--;
			fitness(_it, maxValue);
		}
		else
		{
			_it--;
			x0 = (*_it).objectiveVector()[0];
			y0 = (*_it).objectiveVector()[1];
			_it--;
			x1 = (*_it).objectiveVector()[0];
			_it++;
			_it++;
			y1 = (*_it).objectiveVector()[1];
			_it--;
			fit = (x1 - x0) * (y1 - y0);
			fitness(_it, fit);
			//tmp = (MOEOT*)&(*_it);
			//tmp->fitness(fit);
		}
	}

	double computeTmp(const ObjectiveVector & _objVec, int _where)
	{
		double res, tmp;
		if(hint==begin() || hint==end())
			res=maxValue;
		else{
			if(_where==0){
				//on calcule la fit de celui à potentiellement inserer
				res= (*hint).objectiveVector()[1] - _objVec[1];
				hint--;
				res*= ((*hint).objectiveVector()[0] - _objVec[0]);
				hint++;
			}
			else if(_where <0){
				// on calcule la fit de son predecesseur
				res=  _objVec[1] - (*hint).objectiveVector()[1];
				tmp=(*hint).objectiveVector()[0];
				hint--;
				res*= ((*hint).objectiveVector()[0] - tmp);
				hint++;
			}
			else{
				// on calcule la fit de son successeur
				res= _objVec[0] - (*hint).objectiveVector()[0];
				tmp=(*hint).objectiveVector()[1];
				hint++;
				res*= ((*hint).objectiveVector()[1] - tmp);
				hint--;
			}
		}
		return res;
	}


	void filterbis()
	{
		Iterator it, itd;
		//used to find sol with minimum fitness value
		double minFit = maxValue;

		// remove MOEOT with the lowest fitness value while archive size > maxSize
		while (size() > maxSize)
		{
			//find sol with minimum fitness
			for(it=begin(); it!=end(); it++)
			{
				if(it->fitness() < minFit)
				{
					minFit = it->fitness();
					itd = it;
				}
			}
			//remove it and recompute fitness of its neighbors
			it = itd;
			it--;
			erase(itd);
			compute(it);
			it++;
			compute(it);
		}
	}

	/**
	 * iteratively removes the less-contributing solution from the acrhive
	 */
	bool filter(const MOEOT & _moeo)
	{
		bool res;
		double x, y, pred, succ, tmp=0;

		if(hint==begin() || hint==end())
		{
			insert(hint, _moeo);
			hint--;
			computeFitness(hint);
			filterbis();
			res=true;
		}
		else
		{
			//compute fitness tmp
			tmp=computeTmp(_moeo.objectiveVector(), 0);
			hint--;
			pred=computeTmp(_moeo.objectiveVector(), -1);
			hint++;
			succ=computeTmp(_moeo.objectiveVector(), 1);
			if(tmp>succ || tmp>pred)
			{
				insert(hint, _moeo);
				hint--;
				//ici faudrait utiliser les valeurs qu'on vient de calculer pour les affecter direct (faire attention à ou on se trouve)
				computeFitness(hint);
				filterbis();
				res=true;
			}
			else
			{
				Iterator it;
				double minFit = maxValue;
				for(it=begin(); it!=end(); it++)
				{
					if(it->fitness() < minFit)
					{
						minFit = it->fitness();
					}
				}
				if (tmp<=minFit)
				{
					res = false;
				}
				else
				{
					// REDONDANT arranger le code
					insert(hint, _moeo);
					hint--;
					//ici faudrait utiliser les valeurs qu'on vient de calculer pour les affecter direct (faire attention à ou on se trouve)
					computeFitness(hint);
					filterbis();
					res=true;
				}
			}
		}
		return res;
	}

};

#endif /* MOEO2DMINHYPERVOLUMEARCHIVE_H_ */
