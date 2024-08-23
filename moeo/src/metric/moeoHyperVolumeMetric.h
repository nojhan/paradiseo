/*
* <moeoHyperVolumeMetric.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Jeremie Humeau
* Arnaud Liefooghe
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

#ifndef MOEOHYPERVOLUMEMETRIC_H_
#define MOEOHYPERVOLUMEMETRIC_H_

#include <metric/moeoMetric.h>

/**
 * The hypervolume metric evaluates the multi-dimensional area (hypervolume) enclosed by set of objective vectors and a reference point
 * (E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms: A comparative case study and the strength pareto approach. IEEE Transactions on Evolutionary Computation, 3(4):257–271, 1999)
 */
template < class ObjectiveVector >
class moeoHyperVolumeMetric : public moeoVectorUnaryMetric < ObjectiveVector , double >
  {
  public:

    /**
     * Constructor with a coefficient (rho)
     * @param _normalize allow to normalize data (default true)
     * @param _rho coefficient to determine the reference point.
     */
    moeoHyperVolumeMetric(bool _normalize=true, double _rho=1.1): normalize(_normalize), rho(_rho), ref_point(NULL){
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }

    /**
     * Constructor with a reference point
     * @param _normalize allow to normalize data (default true)
     * @param _ref_point the reference point
     */
    moeoHyperVolumeMetric(bool _normalize=true, ObjectiveVector& _ref_point=NULL): normalize(_normalize), rho(0.0), ref_point(_ref_point){
	    bounds.resize(ObjectiveVector::Traits::nObjectives());
	    // initialize bounds in case someone does not want to use them
	    for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
	    {
	        bounds[i] = eoRealInterval(0,1);
	    }
    }

    /**
     * Constructor with a reference point
     * @param _ref_point the reference point
     * @param _bounds bounds value
     */
    moeoHyperVolumeMetric(ObjectiveVector& _ref_point, std::vector < eoRealInterval >& _bounds): normalize(false), rho(0.0), ref_point(_ref_point), bounds(_bounds){}

    /**
     * calculates and returns the HyperVolume value of a pareto front
     * @param _set the vector contains all objective Vector of pareto front
     */
    double operator()(const std::vector < ObjectiveVector > & _set)
    {
    	std::vector < std::vector<double> > front;

    	//determine the reference point if a coefficient is passed in paremeter
    	if(rho >= 1.0){
    		//determine bounds
    		setup(_set);
    		//determine reference point
       		for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++){
       			if(normalize){
       				if (ObjectiveVector::Traits::minimizing(i))
       					ref_point[i]= rho;
       				else
       					ref_point[i]= 1-rho;
       			}
       			else{
       				if (ObjectiveVector::Traits::minimizing(i))
       					ref_point[i]= bounds[i].maximum() * rho;
       				else
       					ref_point[i]= bounds[i].maximum() * (1-rho);
       			}
        	}
        	//if no normalization, reinit bounds to O..1 for
        	if(!normalize)
         		for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
            		bounds[i] = eoRealInterval(0,1);

    	}
    	else if(normalize)
    		setup(_set);
    	front.resize(_set.size());
    	for(unsigned int i=0; i < _set.size(); i++){
    		front[i].resize(ObjectiveVector::Traits::nObjectives());
	    	for (unsigned int j=0; j<ObjectiveVector::Traits::nObjectives(); j++){
	    		if (ObjectiveVector::Traits::minimizing(j)){
	    			front[i][j]=ref_point[j] - ((_set[i][j] - bounds[j].minimum()) /bounds[j].range());
	    		}
	    		else{
	    			front[i][j]=((_set[i][j] - bounds[j].minimum()) /bounds[j].range()) - ref_point[j];
	    		}
	    	}
    	}

    	return calc_hypervolume(front, front.size(),ObjectiveVector::Traits::nObjectives());
    }

    /**
     * getter on bounds
     * @return bounds
     */
    std::vector < eoRealInterval > getBounds(){
        return bounds;
    }

    /**
     * method caclulate bounds for the normalization
     * @param _set the vector of objective vectors
     */
    void setup(const std::vector < ObjectiveVector > & _set){
    	if(_set.size() < 1)
    		throw("Error in moeoHyperVolumeUnaryMetric::setup -> argument1: vector<ObjectiveVector> size must be greater than 0");
    	else{
	        typename ObjectiveVector::Type min, max;
	        unsigned int nbObj=ObjectiveVector::Traits::nObjectives();
	        bounds.resize(nbObj);
	        for (unsigned int i=0; i<nbObj; i++){
	            min = _set[0][i];
	            max = _set[0][i];
	            for (unsigned int j=1; j<_set.size(); j++){
	                min = std::min(min, _set[j][i]);
	                max = std::max(max, _set[j][i]);
	            }
	            bounds[i] = eoRealInterval(min, max);
	        }
    	}
    }

    /**
     * method calculate if a point dominates another one regarding the x first objective
     * @param _point1 a vector of distances
     * @param _point2 a vector of distances
     * @param _no_objectives a number of objectives
     * @return true if '_point1' dominates '_point2' with respect to the first 'no_objectives' objectives
     */
    bool dominates(std::vector<double>& _point1, std::vector<double>& _point2, unsigned int _no_objectives){
    	unsigned int i;
    	bool better_in_any_objective = false;
    	bool worse_in_any_objective = false;

    	for(i=0; i < _no_objectives && !worse_in_any_objective; i++){
    		if(_point1[i] > _point2[i])
    			better_in_any_objective = true;
    		else if(_point1[i] < _point2[i])
    			worse_in_any_objective = true;
    	}
    	//_point1 dominates _point2 if it is better than _point2 on a objective and if it is never worse in any other objectives
    	return(!worse_in_any_objective && better_in_any_objective);

    }

    /**
     * swap two elements of a vector
     * @param _front the vector
     * @param _i index of the first element to swap
     * @param _j index of the second element to swap
     */
    void swap(std::vector< std::vector<double> >& _front, unsigned int _i, unsigned int _j){
    	std::vector<double> tmp;
    	tmp=_front[_i];
    	_front[_i]=_front[_j];
    	_front[_j]=tmp;
//another way (don't work on visual studio)
//    	_front.push_back(_front[_i]);
//    	_front[_i]= _front[_j];
//    	_front[_j]=_front.back();
//    	_front.pop_back();
    }


    /**
     * collect all nondominated points regarding the first '_no_objectives' objectives (dominated points are stored at the end of _front)
     * @param _front the front
     * @param _no_points the number of points of the front to consider (index 0 to _no_points are considered)
     * @param _no_objectives the number of objective to consider
     * @return the index of the last nondominated point
     */
    unsigned int filter_nondominated_set( std::vector < std::vector< double > >& _front, unsigned int _no_points, unsigned int _no_objectives){
    	unsigned int i,j,n;

    	n=_no_points;
    	i=0;
    	while(i < n){
    		j=i+1;
    		while(j < n){
    			//if a point 'A' (index i) dominates another one 'B' (index j), swap 'B' with the point of index n-1
    			if( dominates(_front[i], _front[j], _no_objectives)){
    				n--;
    				swap(_front, j, n);
    			}
    			//if a point 'B'(index j) dominates another one 'A' (index i), swap 'A' with the point of index n-1
    			else if( dominates(_front[j], _front[i], _no_objectives)){
    				n--;
    				swap(_front, i, n);
    				i--;
    				break;
    			}
    			else
    				j++;
    		}
    		i++;
    	}
    	return n;
    }

    /**
     * find a minimum value
     * @param _front the front
     * @param _no_points the number of points of the front to consider (index 0 to _no_points are considered)
     * @param _objective the objective to consider
     * @return the minimum value regarding dimension '_objective' consider points O to _no_points in '_front'
     */
    double surface_unchanged_to(std::vector < std::vector< double > >& _front, unsigned int _no_points, unsigned int _objective){
    	unsigned int i;
    	double min, value;

    	if(_no_points < 1)
    		throw("Error in moeoHyperVolumeUnaryMetric::surface_unchanged_to -> argument2: _no_points must be greater than 0");
    	min = _front[0][_objective];

    	for(i=1; i < _no_points; i++){
    		value = _front[i][_objective];
    		if(value < min)
    			min = value;
    	}
    	return min;
    }


    /**
     * remove all points having a value <= 'threshold' regarding the dimension 'objective', only points of index 0 to _no_points are considered.
     * points removed are swap at the end of the front.
     * @param _front the front
     * @param _no_points the number of points of the front to consider (index 0 to _no_points are considered)
     * @param _objective the objective to consider
     * @param _threshold the threshold
     * @return index of the last points of '_front' greater than the threshold
     */
    unsigned int reduce_nondominated_set(std::vector < std::vector< double > >& _front, unsigned int _no_points, unsigned int _objective, double _threshold){
    	unsigned int i,n ;

    	n=_no_points;
    	for(i=0; i < n ; i++)
    		if(_front[i][_objective] <= _threshold){
    			n--;
    			swap(_front, i, n);
    			i--; //ATTENTION I had this to reconsider the point copied to index i (it can be useless verify algorythimic in calc_hypervolume)
    		}

    	return n;
    }


    /**
     * calculate hypervolume of the front (data are redrafted before)
     * @param _front the front
     * @param _no_points the number of points of the front to consider (index 0 to _no_points are considered)
     * @param _no_objectives the number of objective to consider
     * @return the hypervolume of the front
     */
    double calc_hypervolume(std::vector < std::vector< double > >& _front, unsigned int _no_points, unsigned int _no_objectives){
    	unsigned int n;
    	double volume, distance;

    	volume=0;
    	distance=0;
    	n=_no_points;
    	while(n > 0){
    		unsigned int no_nondominated_points;
    		double temp_vol, temp_dist;

    		//get back the index of non dominated points of the front regarding the first "_nb_objectives - 1" objectives
    		//So one dimension is not determinante for the dominance
    		no_nondominated_points = filter_nondominated_set(_front, n, _no_objectives - 1);

    		temp_vol=0;

    		//if there are less than 3 objectifs take the fisrt objectif of the first point of front to begin computation of hypervolume
    		if(_no_objectives < 3){
    	    	if(_no_objectives < 1) {
    	    		throw("Error in moeoHyperVolumeUnaryMetric::calc_hypervolume -> argument3: _no_objectives must be greater than 0"); }
    			temp_vol=_front[0][0];
    		}
    		//else if there at least 3 objectives, a recursive computation of hypervolume starts with _no_objectives -1 on the filter_nondominated_set calculating previously.
    		else
    			temp_vol= calc_hypervolume(_front, no_nondominated_points, _no_objectives - 1);

    		//search the next minimum distance on the dimension _no_objectives -1
    		temp_dist = surface_unchanged_to(_front, n, _no_objectives - 1);
    		//calculate the area
    		volume+= temp_vol * (temp_dist - distance);
    		//change distance to have the good lenght on next step
    		distance= temp_dist;
    		//remove all points <= distance on dimension _no_objectives
    		n=reduce_nondominated_set(_front, n , _no_objectives - 1, distance);
    	}
    	return volume;
    }



  private:

	    /*boolean indicates if data must be normalized or not*/
	    bool normalize;

	    double rho;

	    ObjectiveVector ref_point;

	    /*vectors contains bounds for normalization*/
	    std::vector < eoRealInterval > bounds;



  };

#endif /*MOEOHYPERVOLUMEMETRIC_H_*/
