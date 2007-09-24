// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFrontByFrontSharingDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFRONTBYFRONTSHARINGDIVERSITYASSIGNMENT_H_
#define MOEOFRONTBYFRONTSHARINGDIVERSITYASSIGNMENT_H_

#include <diversity/moeoSharingDiversityAssignment.h>

/**
 * Sharing assignment scheme on the way it is used in NSGA.
 */
template < class MOEOT >
class moeoFrontByFrontSharingDiversityAssignment : public moeoSharingDiversityAssignment < MOEOT >
{
public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor
     * @param _distance the distance used to compute the neighborhood of solutions (can be related to the decision space or the objective space)
     * @param _nicheSize neighborhood size in terms of radius distance (closely related to the way the distances are computed)
     * @param _alpha parameter used to regulate the shape of the sharing function
     */
    moeoFrontByFrontSharingDiversityAssignment(moeoDistance<MOEOT,double> & _distance, double _nicheSize = 0.5, double _alpha = 2.0) : moeoSharingDiversityAssignment < MOEOT >(_distance, _nicheSize, _alpha)
    {}


    /**
     * Ctor with an euclidean distance (with normalized objective values) in the objective space is used as default
     * @param _nicheSize neighborhood size in terms of radius distance (closely related to the way the distances are computed)
     * @param _alpha parameter used to regulate the shape of the sharing function
     */
    moeoFrontByFrontSharingDiversityAssignment(double _nicheSize = 0.5, double _alpha = 2.0) : moeoSharingDiversityAssignment < MOEOT >(_nicheSize, _alpha)
    {}


    /**
     * @warning NOT IMPLEMENTED, DO NOTHING !
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     * @warning NOT IMPLEMENTED, DO NOTHING !
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        std::cout << "WARNING : updateByDeleting not implemented in moeoSharingDiversityAssignment" << std::endl;
    }


private:

    using moeoSharingDiversityAssignment < MOEOT >::distance;
    using moeoSharingDiversityAssignment < MOEOT >::nicheSize;
    using moeoSharingDiversityAssignment < MOEOT >::sh;
    using moeoSharingDiversityAssignment < MOEOT >::operator();


    /**
     * Sets similarities FRONT BY FRONT for every solution contained in the population _pop
     * @param _pop the population
     */
    void setSimilarities(eoPop < MOEOT > & _pop)
    {
        // compute distances between every individuals
        moeoDistanceMatrix < MOEOT , double > dMatrix (_pop.size(), distance);
        dMatrix(_pop);
        // sets the distance to bigger than the niche size for every couple of solutions that do not belong to the same front
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            for (unsigned int j=0; j<i; j++)
            {
                if (_pop[i].fitness() != _pop[j].fitness())
                {
                    dMatrix[i][j] = nicheSize;
                    dMatrix[j][i] = nicheSize;
                }
            }
        }
        // compute similarities
        double sum;
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            sum = 0.0;
            for (unsigned int j=0; j<_pop.size(); j++)
            {
                sum += sh(dMatrix[i][j]);
            }
            _pop[i].diversity(sum);
        }
    }

};

#endif /*MOEOFRONTBYFRONTSHARINGDIVERSITYASSIGNMENT_H_*/
