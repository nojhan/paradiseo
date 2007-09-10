#ifndef eoEpsilonArchive_h
#define eoEpsilonArchive_h

#include <moo/eoMOFitness.h>
#include <list>

template <class EOT> 
class eoEpsilonArchive {
    
    typedef typename EOT::Fitness::fitness_traits Traits;
    
    struct Node {
        EOT element;
        std::vector<long> discretized;

        Node(const EOT& eo) : element(eo), discretized(Traits::nObjectives()) {}
    
        dominance::dominance_result check(const Node& other) const {
            return dominance::check_discrete(discretized, other.discretized);
        }
    };

    typedef std::vector<Node> archive_t;

    archive_t archive;
    std::vector<double> inv_eps;
   
    unsigned max_size;
    
    public:
    
    static double direction(unsigned i) { return 1; }
    static double tol() { return 1e-6; }

            
    eoEpsilonArchive(const std::vector<double>& eps_, unsigned max_size_ = std::numeric_limits<unsigned>::max()) : max_size(max_size_) {
        inv_eps.resize(eps_.size());
        for (unsigned i = 0; i < inv_eps.size(); ++i) {
            inv_eps[i] = 1.0 / eps_[i];
        }
        if (inv_eps.size() != Traits::nObjectives()) throw std::logic_error("eoEpsilonArchive: need one epsilon for each objective");    
    }

    bool empty() { return archive.size() == 0; }

    void operator()(const EOT& eo) {
        
        using std::cout;
        using std::endl;
        
        // discretize
        Node node(eo);
        for (unsigned i = 0; i < eo.fitness().size(); ++i) {
            double val = Traits::direction(i) * eo.fitness()[i];
            
            node.discretized[i] = (long) floor(val*inv_eps[i]);
        }

        using namespace dominance;

        unsigned box = archive.size();
        // update archive
        

        for (unsigned i = 0; i != archive.size(); ++i) {
            dominance_result result = node.check(archive[i]); //check<eoEpsilonArchive<EOT> >(node.discretized, archive[i].discretized);

            switch (result) {
                case first : {                          // remove dominated archive member
                    std::swap( archive[i], archive.back());
                    archive.pop_back();
                    --i;
                    break;
                }
                case second : {
                    return;                   // new one does not belong in archive
                }
                case non_dominated : break;             // both non-dominated, put in different boxes
                case non_dominated_equal : {            // in same box
                    box = i;
                    goto exitLoop; // break            
                }
            }

        }

exitLoop:
        // insert 
        if (box >= archive.size()) { // non-dominated, new box
            archive.push_back(node);
        } else { // fight it out
            int dom = node.element.fitness().check_dominance( archive[box].element.fitness() );
            
            switch (dom) {
                case 1: archive[box] = node; break;
                case -1: break;
                case 0: {
                        double d1 = 0.0;
                        double d2 = 0.0;
                        
                        for (unsigned i = 0; i < node.element.fitness().size(); ++i) {
                            double a = Traits::direction(i) * node.element.fitness()[i] * inv_eps[i];
                            double b = Traits::direction(i) * archive[box].element.fitness()[i] * inv_eps[i];
                            
                            d1 += pow( a - node.discretized[i], 2.0);
                            d2 += pow( b - node.discretized[i], 2.0);
                        }
                        
                        if (d1 > d2) {
                            archive[box] = node; 
                        }
                        break;
                    }
            }
        }
        
        /*
        static unsigned counter = 0;
        if (++counter % 500 == 0) {
            std::vector<long> mins(archive[0].discretized.size(), std::numeric_limits<long>::max());
            std::vector<long> maxs(archive[0].discretized.size(), std::numeric_limits<long>::min());
            for (unsigned i = 0; i < archive.size(); ++i) {
                for (unsigned dim = 0; dim < archive[i].discretized.size(); ++dim) {
                    mins[dim] = std::min( mins[dim], archive[i].discretized[dim] );
                    maxs[dim] = std::max( maxs[dim], archive[i].discretized[dim] );
                }
            }
            
            std::cout << "Range ";
            for (unsigned dim = 0; dim < mins.size(); ++dim) {
                std::cout << (maxs[dim] - mins[dim]) << ' ';
            }
            std::cout << archive.size() << std::endl;

        }*/

        if (archive.size() > max_size) {
            unsigned idx = rng.random(archive.size());
            if (idx != archive.size()-1) std::swap(archive[idx], archive.back());
            archive.pop_back();
        }

    }

    void appendTo(eoPop<EOT>& pop) const {
        for (typename archive_t::const_iterator it = archive.begin(); it != archive.end(); ++it) {
            pop.push_back( it->element );
        }
    }

    const EOT& selectRandom() const {
        int i = rng.random(archive.size());
        return archive[i].element;
    }

};


#endif

