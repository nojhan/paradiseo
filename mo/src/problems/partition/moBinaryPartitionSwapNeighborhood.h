#pragma once

#include <utility>

#include <mo>
#include "moBinaryPartition.h"

/** Stable neighborhood for binary partitions.
 *
 * This generates all neighbors of a binary partition
 * that have the same dimension than the considered solution.
 * I.e. it enumerates all the swaps of two atoms
 * between the selected and rejected sets.
 *
 * The core data structure is two indices:
 * - one for the position within the selected set of a binary partition,
 * - the other for the position within the rejected set.
 *
 * The neighborhood is defined as enumerating the neighbors,
 * first by going over the rejected atoms (outer loop),
 * then by iterating over the selected atoms (inner loop).
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moBinaryPartitionSwapNeighborhood : public moNeighborhood<moBinaryPartitionSwapNeighbor<EOT, Fitness> >
{
    public:
        /** Shortcut for neighbor's type. */
        using Neighbor = moBinaryPartitionSwapNeighbor<EOT, Fitness>;

        /** Shortcut for Atom’s type. */
        using AtomType = typename EOT::AtomType;

        /** Get the currently pointed selected atom. */
        AtomType selected(EOT& from, const size_t i_select) {
            typename EOT::ContainerType::iterator
                it = std::begin(from.rejected);
            std::advance(it, i_select);
            return *it;
        }

        /** Get the currently pointed rejected atom. */
        AtomType rejected(EOT& from, const size_t j_reject) {
            typename EOT::ContainerType::iterator
                it = std::begin(from.selected);
            std::advance(it, j_reject);
            return *it;
        }

        /** Initialize the neighborhood.
         *
         * This actually make the neighborhood point to the first possible swap:
         * between the first selected atom and the first rejected atom.
         */
        virtual void init(EOT& from, Neighbor& to) override {
            i_select = 0;
            j_reject = 0;

            // std::clog << "Init neighborhood:"
            //     << " -" << rejected(from, j_reject)
            //     << " +" << selected(from, i_select)
            //     << std::endl;

            // First item in both lists.
            AtomType in  = selected(from, i_select);
            AtomType out = rejected(from, j_reject);
            to.set(in, out);
        }

        /** Point to the next neighbor. */
        virtual void next(EOT& from, Neighbor& to) override {
            // If last item of the inner loop.
            if( i_select == from.rejected.size()-1 ) {
                i_select = 0; // Reset inner loop.
                j_reject++; // Next outer loop.
            } else {
                i_select++; // Next inner loop.
            }

            // std::clog << "Next in neighborhood:"
            //     << " -" << rejected(from, j_reject)
            //     << " +" << selected(from, i_select)
            //     << std::endl;

            assert( from.rejected.contains(selected(from,i_select)) );
            assert( from.selected.contains(rejected(from,j_reject)) );
            assert( selected(from,i_select) != rejected(from,j_reject) );

            // Implant this move in the neighbor.
            to.set(
                selected(from, i_select),
                rejected(from, j_reject)
            );
        }

        /** Returns true if there is more neighbors to be enumerated. */
        virtual bool cont(EOT& from) override {
            // std::clog << "cont neighborhood?"
            //     << " " << j_reject << "(-" << rejected(from, j_reject) << ")/" << from.selected.size()
            //     << " " << i_select << "(-" << selected(from, i_select) << ")/" << from.rejected.size()
            //     << std::endl;

            // If reached the last item of the outer loop.
            if( i_select == from.rejected.size()-1
            and j_reject == from.selected.size()-1) {
                // We should also have reached the end of the inner loop,
                // and have set the inner loop to zero.
                // std::clog << "\tnope" << std::endl;
                return false;

            } else { // There is still some items in the outer loop.
                     // and thus also in the inner loop.
                // std::clog << "\tyes" << std::endl;
                assert( j_reject < from.selected.size() );
                return true;
            }
        }

    /** Returns true if there is actual neighbors in the neighborhood.
     *
     * Essentially just tells if the rejected set is not empty.
     */
    virtual bool hasNeighbor(EOT& solution) override {
        return solution.rejected.size() > 0;
    }

    virtual std::string className() const override {
        return "moBinaryPartitionSwapNeighborhood";
    }

#ifndef NDEBUG
    public:
#else
    protected:
#endif
        /** Index of the currently pointed selected atom. */
        size_t i_select;

        /** Index of the currently pointed rejected atom. */
        size_t j_reject;
};
