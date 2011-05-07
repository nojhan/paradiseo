// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// PO.h
// (c) OPAC 2007
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: thomas.legrand@lifl.fr
 */
//-----------------------------------------------------------------------------

#ifndef PO_H
#define PO_H

//-----------------------------------------------------------------------------
#include <stdexcept>
#include <EO.h>
//-----------------------------------------------------------------------------

/** PO inheriting from EO is specially designed for particle swarm optimization particle.POs have got a fitness,
    which at the same time needs to be only an object with the operation less than (<)
    defined. A best fitness also belongs to the particle.Fitness says how
    good is the particle for a current iteration whereas the best fitness can be saved for
    many iterations.

    @ingroup Core
*/
template < class F > class PO:public EO < F >
{

public:

        #if defined(__CUDACC__)
                typedef typename EO < F >::Fitness Fitness;
        #else
                typedef typename PO<F>::Fitness Fitness;
        #endif

    /** Default constructor.
        Fitness must have a ctor which takes 0 as a value. Best fitness mush also have the same constructor.
    */
    PO ():repFitness (Fitness ()), invalidFitness (true),
            bestFitness (Fitness()){}


    /// Return fitness value.
    Fitness fitness () const
    {
        if (invalid ())
            throw std::runtime_error ("invalid fitness in PO.h");
        return repFitness;
    }


    /** Set fitness. At the same time, validates it.
    *  @param _fitness New fitness value.
    */
    void fitness (const Fitness & _fitness)
    {
        repFitness = _fitness;
        invalidFitness = false;
    }

    /** Return the best fitness.
    * @return bestFitness
    */
    Fitness best () const
    {
        if (invalid ())
            throw std::runtime_error ("invalid best fitness in PO.h");
        return bestFitness;
    }


    /** Set the best fitness.
    *  @param _bestFitness New best fitness found for the particle.
    */
    void best (const Fitness & _bestFitness)
    {
        bestFitness = _bestFitness;
        invalidBestFitness = false;
    }


    /** Return true If fitness value is invalid, false otherwise.
     *  @return true If fitness is invalid.
     */
    bool invalid () const
    {
        return invalidFitness;
    }

    /**  Invalidate the fitness.
    * @return
    */
    void invalidate ()
    {
        invalidFitness = true;
    }

    /** Return true If the best fitness value is invalid, false otherwise.
     *  @return true If the bestfitness is invalid.
     */
    bool invalidBest () const
    {
        return invalidBestFitness;
    }

    /**  Invalidate the best fitness.
    * @return
    */
    void invalidateBest ()
    {
        invalidBestFitness = true;
    }

    /** Return the class id.
     *  @return the class name as a std::string
     */
    virtual std::string className () const
    {
        return "PO";
    }

    /** Returns true if
        @return true if the fitness is higher
    */
    bool operator< (const PO & _po2) const { return fitness () < _po2.fitness ();}
    bool operator> (const PO & _po2) const { return !(fitness () <= _po2.fitness ());}


     /**
      * Write object. Called printOn since it prints the object _on_ a stream.
      * @param _os A std::ostream.
      */
    virtual void printOn(std::ostream& _os) const { _os << bestFitness << ' ' ;}


    /**
     * Read object.\\
     * Calls base class, just in case that one had something to do.
     * The read and print methods should be compatible and have the same format.
     * In principle, format is "plain": they just print a number
     * @param _is a std::istream.
     * @throw runtime_std::exception If a valid object can't be read.
     */
    virtual void readFrom(std::istream& _is) {

        // the new version of the reafFrom function.
        // It can distinguish between valid and invalid fitness values.
        std::string fitness_str;
        int pos = _is.tellg();
        _is >> fitness_str;

        if (fitness_str == "INVALID")
        {
            invalidFitness = true;
        }
        else
        {
            invalidFitness = false;
            _is.seekg(pos); // rewind
            _is >> repFitness;
        }
    }

private:
    Fitness repFitness;		// value of fitness for this particle
    bool invalidFitness;		// true if the value of the fitness is invalid

    Fitness bestFitness;		// value of the best fitness found for the particle
    bool invalidBestFitness;	// true if the value of the best fitness is invalid

};

//-----------------------------------------------------------------------------

#endif /*PO_H */
