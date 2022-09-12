#pragma once

#include <set>

#include <eo>

template<class FitT>
class moBinaryPartition : public EO<FitT>
{
    public:
        using AtomType = size_t;
        using ContainerType = std::set<AtomType>;

        ContainerType selected;
        ContainerType rejected;

        /** Constructor
         *
         * @param total_nb_genes Total number of possible genes from whith to select.
         */
        moBinaryPartition( const size_t total_nb_genes )
        {
            // Fill the rejected list with all possible gene indices,
            // starting from zero.
            for(size_t i = 0; i < total_nb_genes; ++i) {
                rejected.insert(i);
            }
            // No selected.
        }

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

        /** Serialization of the `selected` atoms. */
        virtual void printOn(std::ostream& out) const
        {
            EO<FitT>::printOn(out); // Fitness.
            // Trailing space already inserted.
            out << selected.size() << "  "; // Size.
            std::copy(std::begin(selected), std::end(selected),
                std::ostream_iterator<AtomType>(out, " ")); // Values.
            out << "   ";
            out << rejected.size() << "  "; // Size.
            std::copy(std::begin(rejected), std::end(rejected),
                std::ostream_iterator<AtomType>(out, " ")); // Values.
        }

        /** Deserialization of the `selected` atoms. */
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

        bool operator==(const moBinaryPartition& other) {
            return this->selected == other.selected
               and this->rejected == other.rejected;
        }

        virtual std::string className() const
        {
            return "moBinaryPartition";
        }
};
