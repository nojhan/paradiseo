#pragma once

#include <set>

#include <eo>

/** A partition of a binary space.
 *
 * This data structure defines a grouping of the elements of a multi-dimensional
 * set in a space of boolean numbers.
 * \f[
 *      \mathrm{1}^n = \bigcup_{i=1}^n \{0,1\}_i
 * \f]
 * Elements of the set may be either "selected" (in the set S) or "rejected" (in the set R).
 * \f[
 *      (S \in \mathrm{1}^m) \cup (R \in \mathrm{1}^k) \in \mathrm{1}^n,\; n=m+k
 * \f]
 * Elements are referred to by their index in the set (hereby named "atoms").
 *
 * This representation is useful if your problem can be defined has selecting
 * a subset of elements that optimize some objective function.
 *
 * The core data structures are two ordered sets of unique atoms,
 * the union of which is guaranteed to have the correct dimension.
 */
template<class FitT>
class moBinaryPartition : public EO<FitT>
{
    public:
        /** The type for indices. */
        using AtomType = size_t;

        /** The data structures holding the indices. */
        using ContainerType = std::set<AtomType>;

        /** The set of selected atoms. */
        ContainerType selected;

        /** The set of not-selected atoms. */
        ContainerType rejected;

        /** Consistent constructor
         *
         * Put all `total_nb_atoms` indices in the @ref rejected set.
         * Indices starts at zero and fill the set in increasing order.
         *
         * @param total_nb_atoms Total number of possible atoms from whith to select.
         */
        moBinaryPartition( const size_t total_nb_atoms )
        {
            // Fill the rejected list with all possible gene indices,
            // starting from zero.
            for(size_t i = 0; i < total_nb_atoms; ++i) {
                rejected.insert(i);
            }
            // None selected.
        }

        /** Empty constructor
         *
         * Do not fill the @ref rejected set.
         * You are responsible for making it consistent after instantiation.
         *
         * @warning If you do not fill at least the @ref rejected set,
         *          errors will be raised whe trying to @ref select or @ref reject.
         */
        moBinaryPartition()
        { }

        /** Move one atom in the @ref selected set.
         *
         * That is: erase the atom from @ref rejected,
         * insert it in @ref selected.
         *
         * @note In debug mode, double check that elements were actually moved.
         */
        void select(const size_t atom) {
            assert(not selected.contains(atom));

            #ifndef NDEBUG
                size_t has_erased =
            #endif
            this->rejected.erase(atom);
            assert(has_erased == 1);

            #ifndef NDEBUG
                auto [where, has_inserted] =
            #endif
            this->selected.insert(atom);
            assert(has_inserted);
        }

        /** Move one atom in the @ref rejected set.
         *
         * That is: insert the atom in @ref rejected,
         * erase it from @ref selected.
         *
         * @note In debug mode, double check that elements were actually moved.
         */
        void reject(const size_t atom) {
            assert(not rejected.contains(atom));

            #ifndef NDEBUG
                size_t has_erased =
            #endif
            this->selected.erase(atom);
            assert(has_erased == 1);

            #ifndef NDEBUG
                auto [where, has_inserted] =
            #endif
            this->rejected.insert(atom);
            assert(has_inserted);
        }

        /** Serialization of the `selected` and `rejected` atoms.
         *
         * Output a string of the form (spaces replaced with period here, to show their count):
         *   `<fitness>..<nb_selected>..<sel_0>…<sel_n>...<nb_rejected>..<rej_0>…<rej_m>`
         */
        virtual void printOn(std::ostream& out) const
        {
            EO<FitT>::printOn(out); // Fitness.
            // Trailing space already inserted.
            out << " " << selected.size() << "  "; // Size.
            std::copy(std::begin(selected), std::end(selected),
                std::ostream_iterator<AtomType>(out, " ")); // Values.
            out << "   ";
            out << rejected.size() << "  "; // Size.
            std::copy(std::begin(rejected), std::end(rejected),
                std::ostream_iterator<AtomType>(out, " ")); // Values.
        }

        /** Deserialization of the `selected` and `rejected` atoms.
         *
         * Expects a string of the form (spaces replaced with period here, to show their count):
         *   `<fitness>..<nb_selected>..<sel_0>…<sel_n>...<nb_rejected>..<rej_0>…<rej_m>`
         */
        virtual void readFrom(std::istream& in)
        {
            EO<FitT>::readFrom(in); // Fitness.
            unsigned size;
            in >> size; // Size.
            for(size_t i = 0; i < size; ++i) {
                AtomType atom;
                in >> atom; // Value.
                selected.insert(atom);
            }
            assert(selected.size() == size);
            in >> size; // Size.
            for(size_t i = 0; i < size; ++i) {
                AtomType atom;
                in >> atom; // Value.
                rejected.insert(atom);
            }
            assert(rejected.size() == size);
        }

        /** Returns true if all sets are equals. */
        bool operator==(const moBinaryPartition& other) {
            return this->selected == other.selected
               and this->rejected == other.rejected;
        }

        virtual std::string className() const
        {
            return "moBinaryPartition";
        }
};
