#pragma once

#include <utility>

#include <mo>

#include "moBinaryPartition.h"

template<class EOT, class Fitness=typename EOT::Fitness>
class moBinaryPartitionSwapNeighbor :
            public moBackableNeighbor<EOT,double>//,
            // public moIndexNeighbor<EOT,double>
{
    public:
        using AtomType = typename EOT::AtomType;
        using ContainerType = typename EOT::ContainerType;
        using moBackableNeighbor<EOT, Fitness>::fitness;
        // using moIndexNeighbor<EOT, Fitness>::key;
        // using moIndexNeighbor<EOT, Fitness>::index;

        moBinaryPartitionSwapNeighbor( const size_t _selected_nb ) :
            selected_nb(_selected_nb),
            is_set(false)
        {
            assert(selected_nb > 0);
        }

        virtual void move(EOT& solution) override {
            assert(is_set);
            // Swap the two atoms.
            solution.reject(this->reject);
            solution.select(this->select);
            assert(solution.selected.size() == this->selected_nb);

            solution.invalidate();
        }

        virtual void moveBack(EOT& solution) override {
            assert(is_set);
            solution.reject(this->select);
            solution.select(this->reject);
            assert(solution.selected.size() == this->selected_nb);

            solution.invalidate();
        }

        void set(AtomType in, AtomType out) {
            this->select = in;
            this->reject = out;
            #ifndef NDEBUG
                is_set = true;
            #endif
        }

        std::pair<AtomType,AtomType> get() {
            assert(is_set);
            return std::make_pair(select, reject);
        }

        virtual bool equals(moBinaryPartitionSwapNeighbor<EOT,Fitness>& neighbor) {
            auto [in, out] = neighbor.get();
            return this->select == in and this->reject == out;
        }

        virtual std::string className() const override {
            return "moBinaryPartitionSwapNeighbor";
        }

        virtual void printOn(std::ostream& out) const override {
            assert(is_set);
            out << selected_nb
                << " -" << reject
                << " +" << select;
        }

#ifndef NDEBUG
    public:
#else
    protected:
#endif
        const size_t selected_nb;
        AtomType select;
        AtomType reject;
        #ifndef NDEBUG
            bool is_set;
        #endif
};


