#pragma once

#include <utility>

#include <mo>
#include "moBinaryPartition.h"

/** Stable neighbor for a binary partition.
 *
 * Models how to move from a solution to a neighbor,
 * by swaping one selected atom for one rejected atom.
 * The number of selected atoms is thus guaranteed to be stable.
 *
 * The core data structure is two atoms:
 * - the selected one,
 * - the rejected one.
 */
template<class EOT, class Fitness=typename EOT::Fitness>
class moBinaryPartitionSwapNeighbor :
            public moBackableNeighbor<EOT,Fitness>//,
            // public moIndexNeighbor<EOT,double> // FIXME see if we can model that.
{
    public:
        /** Shortcut for Atom’s type. */
        using AtomType = typename EOT::AtomType;

        /** Shortcut for container’s type. */
        using ContainerType = typename EOT::ContainerType;

        /** Shortcut for fitness. */
        using moBackableNeighbor<EOT, Fitness>::fitness;

        // using moIndexNeighbor<EOT, Fitness>::key;
        // using moIndexNeighbor<EOT, Fitness>::index;

        /** Consistent constructor.
         *
         * Will ensure that the dimension of the partition does not change.
         *
         * @param _selected_nb Number of selected atoms to maintain.
         */
        moBinaryPartitionSwapNeighbor( const size_t _selected_nb ) :
            selected_nb(_selected_nb)
            #ifndef NDEBUG
                , is_set(false)
            #endif
        {
            assert(selected_nb > 0);
        }

        /** Default constructor.
         *
         * Will NOT ensure that the dimension of the partition does not change.
         */
        moBinaryPartitionSwapNeighbor() :
            selected_nb(0)
            #ifndef NDEBUG
                , is_set(false)
            #endif
        {
            // Invalid fitness by default.
        }

        /** Copy constructor.
         */
        moBinaryPartitionSwapNeighbor( const moBinaryPartitionSwapNeighbor<EOT>& other) :
            selected_nb(other.selected_nb),
            select(other.select),
            reject(other.reject)
            #ifndef NDEBUG
                , is_set(other.is_set)
            #endif
        {
            this->fitness(other.fitness());
        }

        /** Default assignment operator.
         */
        moBinaryPartitionSwapNeighbor<EOT>& operator=(
            const moBinaryPartitionSwapNeighbor<EOT>& other)
        {
            this->selected_nb = other.selected_nb;
            this->select = other.select;
            this->reject = other.reject;
            #ifndef NDEBUG
                this->is_set = other.is_set;
            #endif
            this->fitness(other.fitness());
            return *this;
        }

        /** Apply the currently stored move.
         *
         * That is: reject one atom and select one other.
         */
        virtual void move(EOT& solution) override {
            assert(is_set);
            // Swap the two atoms.
            solution.reject(this->reject);
            solution.select(this->select);
            assert(solution.selected.size() == this->selected_nb);
            // this->fitness( Fitness(solution.fitness()) ); // For the cache.
            solution.invalidate();
        }

        /** Apply the opposite of the currently stored move.
         *
         * That is: reject the selected atom, and select the rejected one.
         */
        virtual void moveBack(EOT& solution) override {
            assert(is_set);
            solution.reject(this->select);
            solution.select(this->reject);
            assert(solution.selected.size() == this->selected_nb);
            // this->fitness( Fitness(solution.fitness()) ); // For the cache.
            solution.invalidate();
        }

        /** Set the considered atoms.
         *
         * @param in The selected atom.
         * @param out The rejected atom.
         */
        void set(AtomType in, AtomType out) {
            this->select = in;
            this->reject = out;
            #ifndef NDEBUG
                is_set = true;
            #endif
            this->invalidate();
        }

        /** Set the considered atoms.
         *
         * @param in_out A pair of {selected,rejected} atoms.
         */
        void set(std::pair<AtomType,AtomType> in_out) {
            this->set(in_out.first, in_out.second);
        }

        /** Get the considered atom.
         *
         * @returns A pair of atoms, the first being the selected atom, the second being the rejected one.
         */
        std::pair<AtomType,AtomType> get() {
            assert(is_set);
            return std::make_pair(select, reject);
        }

        /** Returns true if this neighbor has the same selected & rejected atoms than the given neighbor. */
        virtual bool equals(moBinaryPartitionSwapNeighbor<EOT,Fitness>& neighbor) {
            auto [in, out] = neighbor.get();
            return this->select == in and this->reject == out;
        }
    private:
        // Disable access to `equals(moNeighbor<…>&)` (removes the related overloaded-virtual warning).
        using moBackableNeighbor<EOT,Fitness>::equals;

    public:

        //! Class name for state management.
        virtual std::string className() const override {
            return "moBinaryPartitionSwapNeighbor";
        }

        /** Fancy print. */
        virtual void printOn(std::ostream& out) const override {
            assert(is_set);
            EO<Fitness>::printOn(out); // Fitness.
            out << " "
                << selected_nb
                << " -" << reject
                << " +" << select;
        }

        //! Accessor to set the size.
        void size(size_t _selected_nb) {
            assert(_selected_nb > 0);
            this->selected_nb = _selected_nb;
            this->invalidate();
        }

        //! Accessor to get the size.
        size_t size() const {
            return this->selected_nb;
        }

        // virtual void fitness(const Fitness& fit) override
        // {
        //     // std::clog << "Fitness assignment -- neighbor: " << *this << " gets fitness: " << fit << std::endl;
        //     EO<Fitness>::fitness(fit);
        // }

        // virtual const Fitness& fitness() const override
        // {
        //     return EO<Fitness>::fitness();
        // }

#ifndef NDEBUG
    public:
#else
    protected:
#endif
        /** Fixed dimension of the handled solutions. */
        size_t selected_nb;

        /** Selected atom. */
        AtomType select;

        /** Rejected atom. */
        AtomType reject;

        #ifndef NDEBUG
            /** Sanity flag.
             *
             * Used in debug builds to ensure that the neighbor
             * have been set before being used.
             */
            bool is_set;
        #endif
};


