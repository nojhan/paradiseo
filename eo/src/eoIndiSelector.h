/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoIndiSelector.h
    Abstract selection operator, which is used by the eoGeneralOps
    to obtain individuals from a source population. It also gives a 
    direct descended eoPopIndiSelector that can be used to 
    initialize objects with an eoPop<EOT>. For most uses use eoPopIndividualSelector
    rather than eoIndividualSelector to derive from.

 (c) Maarten Keijzer (mkeijzer@mad.scientist.com) and GeNeura Team, 1999, 2000
 
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */

#ifndef eoIndiSelector_h
#define eoIndiSelector_h

/**
 * eoIndividualSelector: This class defines the interface 
*/
template <class EOT>
class eoIndiSelector
{
public :

    eoIndiSelector() {}

    virtual ~eoIndiSelector(void) {}

    /**
        return the number of individuals that can be selected by an nary operator (through operator[] below)
    */
    virtual size_t size(void) const = 0;

    /**
        return the specified individual, the size_t argument should be between 0 and eoIndiSelector::size()
    */
    virtual const EOT& operator[](size_t i) const = 0;

    /**
        Select an individual, maybe from an underlying population (see eoPopIndiSelector)
    */
    virtual const EOT&   operator()(void) = 0;
    
    /// default implementation just calls operator() a couple of times
    /// this can be overridden in favour of a more efficient implementation
    virtual vector<const EOT*>    select(size_t _how_many)
    {   vector<const EOT*> result(_how_many);

        for (unsigned i = 0; i < _how_many; ++i)
        {
            result[i] = &operator()();
        }

        return result;
    }
};

/**
 * eoPopIndiSelector: Intermediate class for dispensing populations
    various useful things can be done with this class:
        you can specify how many of the population can ever be dispensed to the
        operators, but you can also specify a preference to the first guy being
        dispensed. This is useful if you want to perform the operator on a specific
        individual.
*/
template <class EOT>
class eoPopIndiSelector : public eoIndiSelector<EOT>
{
    public :
        eoPopIndiSelector(void) : eoIndiSelector<EOT>(), pop(0), last(0), firstChoice(-1), secondChoice(-1) {}
        
        virtual ~eoPopIndiSelector(void) {}
        
        struct eoUnitializedException{};

        /** Initialization function, binds the population to the selector, can also
            be used to specify an optional end 
        */
        eoPopIndiSelector& bind(const eoPop<EOT>& _pop, int _end = -1)
        {
            pop = &_pop;
            last = _end;

            if (last < 0 || last > (int) pop->size())
            {
                last = pop->size();
            }

            return *this;
        }

        /** Bias function to bias the selection function to select specific individuals
            first before applying a selection algorithm defined in derived classes
        */
        eoPopIndiSelector& bias(int _first, int _second = -1)
        {
            firstChoice  = _first;
            secondChoice = _second;
            return *this;
        }


        size_t size(void) const { valid(); return last; }
        const EOT& operator[](size_t _i) const { valid(); return pop->operator[](_i); }

        eoPop<EOT>::const_iterator begin(void) const { valid(); return pop->begin(); }
        eoPop<EOT>::const_iterator end(void)   const { valid(); return pop->end(); }


        /// operator() does the work. Note that it is not virtual. It calls do_select that needs to be implemented by the derived classes
        const EOT& operator()(void)
        {
            valid();
            if (firstChoice < 0 || firstChoice >= (int) size())
            {
                // see if we have a second choice
                if (secondChoice < 0 || secondChoice >= (int) size())
                {
                    return do_select(); // let the child figure out what to do  
                }
                    
                const EOT& result = pop->operator[](secondChoice);
                secondChoice = -1;
                return result;
            }
                        
            const EOT& result = pop->operator[](firstChoice);
            firstChoice = -1;
            return result;
        }

        /**
            do_select, abstract member re-implemented by derived classes
            This function will get called by operator() when it ran out of choices
        */
        virtual const EOT& do_select(void) = 0;

    private :

        void valid(void) const
        {
            if (pop == 0)
                throw eoUnitializedException();
        }
  
        const eoPop<EOT>* pop; // need a pointer as this the pop argument can be re-instated
        int   last;
        int   firstChoice;
        int   secondChoice;
};

#endif
