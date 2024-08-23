/*
* <moeoEpsilonHyperboxArchive.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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
// moeoEpsilonHyperboxArchive.h
//-----------------------------------------------------------------------------

#ifndef MOEOEPSILONBOXARCHIVE_H_
#define MOEOEPSILONBOXARCHIVE_H_

#include <eoPop.h>
#include <comparator/moeoComparator.h>
#include <comparator/moeoObjectiveVectorComparator.h>
#include <distance/moeoEuclideanDistance.h>
#include <utils/moeoObjectiveVectorNormalizer.h>
#include <utils/eoRealBounds.h>

/**
 * This class represents an epsilon hyperbox archive.
 */
template < class MOEOT >
class moeoEpsilonHyperboxArchive : public moeoArchive < MOEOT >
{
public:

    using moeoArchive < MOEOT > :: size;
    using moeoArchive < MOEOT > :: resize;
    using moeoArchive < MOEOT > :: operator[];
    using moeoArchive < MOEOT > :: back;
    using moeoArchive < MOEOT > :: pop_back;
    using moeoArchive < MOEOT > :: push_back;
    using moeoArchive < MOEOT > :: begin;
    using moeoArchive < MOEOT > :: end;
    using moeoArchive < MOEOT > :: replace;


    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor where you can choose your own moeoObjectiveVectorComparator
     * @param _comparator the functor used to compare objective vectors
     * @param _epsilon the vector contains epsilon values for each objective
     * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
     */
    moeoEpsilonHyperboxArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, std::vector<double> _epsilon, bool _replace=true) : moeoArchive < MOEOT >(_comparator, _replace), epsilon(_epsilon), bounds(0.0, 1.0), normalizer(bounds, 1.0)
    {}

    /**
      * Default Ctor
      * @param _epsilon the vector contains epsilon values for each objective
      * @param _replace boolean which determine if a solution with the same objectiveVector than another one, can replace it or not
      */
    moeoEpsilonHyperboxArchive(std::vector<double> _epsilon, bool _replace=true) : moeoArchive < MOEOT >(paretoComparator, _replace), epsilon(_epsilon), bounds(0.0, 1.0), normalizer(bounds, 1.0)
    {}

    /**
     * Updates the archive with a given individual _moeo
     * @param _moeo the given individual
     * @return true if _moeo is non-dominated (and not if it is added to the archive)
     */
    bool operator()(const MOEOT & _moeo)
    {
		bool res=false;
		unsigned int  i=0;
		bool nonstop = true;
		bool same = true;
		int  change = 0;
		MOEOT removed;

		//if the archive is empty, we accept automaticaly _moeo
		if(size()==0){
			push_back(_moeo);
			ideal = _moeo.objectiveVector();
			nadir = _moeo.objectiveVector();
			res = true;
		}
		else{
			//change bounds if necessary
			change = changeBounds(_moeo);

			//if change < 0, we have detected that _moeo is bad
			//else there are 4 cases:
			if(change >= 0){
				//calculate the hyperbox corner of _moeo
				ObjectiveVector corner, tmp;
				corner=hyperbox(_moeo);

				//test if _moeo hyperbox corner dominates a hyperbox corner of an element of the archive
				while(nonstop && (i<size())){
					same = true;
					//calculate the hyperbox corner of the ieme element of the archive
					tmp=hyperbox(operator[](i));

					//CASE 1: _moeo epsilon-domine the ieme element of the archive
					if(this->comparator(tmp, corner)){
						std::cout << "ENTER CASE 1" << std::endl;
						//test if bounds changed
						//removed=operator[](i);
						//delete the ieme element of the archive
						if(i==size()-1)
							pop_back();
						else{
							operator[](i)=back();
							pop_back();
							i--;
						}
						//changeBoundsByDeleting(removed);
						res = true;
					}//END CASE 1
					//CASE 2: the ieme element of the archive epsilon-domine _moeo
					else if(this->comparator(corner, tmp)){
						std::cout << "ENTER CASE 2" << std::endl;
						if(change == 1)
							changeBoundsByDeleting(_moeo);
						//we can stop
						nonstop = false;
					}//END CASE 2
					// _moeo is no-epsilon-dominated by archive[i] and arhcive[i] is no-epsilon-dominated by _moeo
					else{
						//test if the hyperbox corner are the same
						for(unsigned int  j=0; j<corner.size(); j++)
							same = same && (corner[j] == tmp[j]);
						//CASE 3: _moeo is in the same hyperbox of archive[i]
						if(same){
							std::cout << "ENTER CASE 3" << std::endl;
							// _moeo dominates archive[i]
							if(this->comparator(operator[](i).objectiveVector(), _moeo.objectiveVector())){
								if(i==size()-1)
									pop_back();
								else{
									operator[](i)=back();
									pop_back();
									i--;
								}
//								removed=operator[](i);
//								operator[](i) = _moeo;
//								changeBoundsByDeleting(removed);
								res=true;
							}
							// _moeo is dominated by archive[i]
							else if(this->comparator(_moeo.objectiveVector(), operator[](i).objectiveVector())){
								changeBoundsByDeleting(_moeo);
								nonstop=false;
							}
							else{
								//keep the one who have the shortest euclidian distance between the corner
								moeoEuclideanDistance < MOEOT > dist;
								double d1 = dist(_moeo.objectiveVector(), corner);
								double d2 = dist(operator[](i).objectiveVector(), corner);
								if(d1 <= d2){
									if(i==size()-1)
										pop_back();
									else{
										operator[](i)=back();
										pop_back();
										i--;
									}
//									removed=operator[](i);
//									operator[](i) = _moeo;
//									changeBoundsByDeleting(removed);
									res=true;
								}
								else{
									nonstop=false;
//									changeBoundsByDeleting(_moeo);
									res=true;
								}
							}

						}//END CASE 3
					}
					i++;
				}
				//CASE 4: _moeo have is place in a empty hyperbox
				if(nonstop){
					std::cout << "ENTER CASE 4" << std::endl;
					push_back(_moeo);
					res=true;
					recalculateBounds();
				}//END CASE 4
			}
			else{
				std::cout << "ENTER CASE 5" << std::endl;
			}
		}

    	return res;
    }

    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     * @return if an archive's element is non-dominated (and not if it is added to the archive)
     */
    bool operator()(const eoPop < MOEOT > & _pop)
    {
    	bool res, tmp = false;
    	for(unsigned int i=0; i<_pop.size(); i++){
    		tmp = (*this)(_pop[i]);
    		res = res || tmp;
    	}
    	return res;
    }


    /**
     * get the nadir point
     * @return ObjectiveVector corresponding to the nadir point
     */
    ObjectiveVector getNadir(){
    	return nadir;
    }

    /**
     * get the idealpoint
     * @return ObjectiveVector corresponding to the ideal point
     */
    ObjectiveVector getIdeal(){
    	return ideal;
    }

    void filtre(){
    	eoPop<MOEOT> pop;
    	for(unsigned i=0; i<size(); i++)
    		pop.push_back(operator[](i));
    	for(unsigned i=0; i<pop.size(); i++)
    		(*this)(pop[i]);
    }


private:

    /**
     * calculate the hyperbox corner of _moeo
     * @param _moeo the given individual
     * @return the ObjectiveVector contains the hyperbox corner values
     */
    ObjectiveVector hyperbox(const MOEOT & _moeo){
    	//normalize _moeo's objectiveVector
    	ObjectiveVector res;
    	res = normalizer(_moeo.objectiveVector());

//    	std::cout << "ObjectiveVector non normalise:"<< _moeo.objectiveVector() << std::endl;
//    	std::cout << "ObjectiveVector normalise:"<< res << std::endl;

    	//calculate the hyperbox corner
    	for(unsigned int  i=0; i<ObjectiveVector::nObjectives(); i++){
    		if(ObjectiveVector::minimizing(i))
    			res[i] = floor(res[i]*1.0/epsilon[i]);
    		else
    			res[i] = ceil(res[i]*1.0/epsilon[i]);
    	}
//    	std::cout << "ObjectiveVector epsilone:" << res << std::endl;
    	return res;
    }

    /**
     * changes ideal and nadir point if _moeo is out of bounds and is not bad
     * @param _moeo the given individual
     * @return if bounds changed or not (1 -> changed, 0 -> not changed, -1 -> _moeo is bad)
     */
    int changeBounds(const MOEOT & _moeo){
//    	std::cout << "changebounds objVec: "<< _moeo.objectiveVector() << std::endl;
    	int  res = 0;
    	//check if an objective is better than the corresponding of the current ideal point
    	for(unsigned int i=0; i<ObjectiveVector::nObjectives(); i++){
    		if(ObjectiveVector::minimizing(i)){
    			if(_moeo.objectiveVector()[i] < ideal[i]){
    				ideal[i]=_moeo.objectiveVector()[i];
    				res = 1;
    			}
    		}
			else{
    			if(_moeo.objectiveVector()[i] > ideal[i]){
    				ideal[i]=_moeo.objectiveVector()[i];
    				res = 1;
    			}
			}
    	}
    	//check if an objective is worst than the corresponding of the current nadir point
		for(unsigned int i=0; i<ObjectiveVector::nObjectives(); i++){
			if(ObjectiveVector::minimizing(i)){
				if(_moeo.objectiveVector()[i] > nadir[i]){
					if(res == 1)
						nadir[i]=_moeo.objectiveVector()[i];
					else
						res = -1; // no objective is better than the ideal and some are worst than nadir -> _moeo is bad
				}
			}
			else{
				if(_moeo.objectiveVector()[i] < nadir[i]){
					if(res == 1)
						nadir[i]=_moeo.objectiveVector()[i];
					else
						res = -1; // no objective is better than the ideal and some are worst than nadir -> _moeo is bad
				}
			}
		}
		//If bounds are changed, change the scale of normalizer
		if(res == 1){
			ObjectiveVector mini;
			ObjectiveVector maxi;
			for(unsigned int i=0; i<ObjectiveVector::nObjectives(); i++){
				mini[i]=std::min(ideal[i], nadir[i]);
				maxi[i]=std::max(ideal[i], nadir[i]);
			}
			normalizer.update_by_min_max(mini, maxi);
		}
//	    std::cout << "change nadir: " << nadir << std::endl;
//	    std::cout << "change ideal: " << ideal << std::endl;
//		std::cout << "res: " << res << std::endl;
    	return res;
    }

    /**
     * when a element is deleting, change the bounds if neccesary.
     * @param _moeo the deleted individual
     */
    void changeBoundsByDeleting(const MOEOT & _moeo){
    	for(unsigned int i=0; i< ObjectiveVector::nObjectives(); i++){
    		if((_moeo.objectiveVector()[i]==nadir[i]) || (_moeo.objectiveVector()[i]==ideal[i]) )
    			return recalculateBounds();
    	}
    }

    /**
     * recalculate ideal and nadir point and change scale of normalizer
     */
    void recalculateBounds(){
    	ObjectiveVector tmp;
		ideal=operator[](0).objectiveVector();
		nadir=operator[](0).objectiveVector();
		if (size() > 1){
			for(unsigned int i=0; i< ObjectiveVector::nObjectives(); i++){
				for(unsigned int j=1; j<size(); j++){
					tmp=operator[](j).objectiveVector();
					if(ObjectiveVector::minimizing(i)){
						if(tmp[i] < ideal[i])
							ideal[i] = tmp[i];
						else if(tmp[i] > nadir[i])
							nadir[i] = tmp[i];
					}
					else{
						if(tmp[i] > ideal[i]){
							ideal[i] = tmp[i];
						}
						else if(tmp[i] < nadir[i]){
							nadir[i] = tmp[i];
						}
					}
				}
			}
		}
		ObjectiveVector mini;
		ObjectiveVector maxi;
		for(unsigned int i=0; i<ObjectiveVector::nObjectives(); i++){
			mini[i]=std::min(ideal[i], nadir[i]);
			maxi[i]=std::max(ideal[i], nadir[i]);
		}
		normalizer.update_by_min_max(mini, maxi);
    }



    /** A moeoObjectiveVectorComparator based on Pareto dominance (used as default) */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;

    /** epsilon values */
    std::vector <double> epsilon;

    /** ideal point of the archive */
    ObjectiveVector ideal;
    /** nadir point of the archive */
    ObjectiveVector nadir;

    /** bounds use by default to initialize the normalizer */
    eoRealInterval bounds;

    /** the objective vector normalizer */
    moeoObjectiveVectorNormalizer <MOEOT> normalizer;

};

#endif /*MOEOEPSILONBOXARCHIVE_H_*/
