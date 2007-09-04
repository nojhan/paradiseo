#ifndef eoEpsilonArchive_h
#define eoEpsilonArchive_h

#include <moo/eoMOFitness.h>
#include <list>

template <class EOT> 
class eoEpsilonArchive {
    
    typedef typename EOT::Fitness::fitness_traits Traits;
    
    struct Node {
        EOT element;
        std::vector<double> discretized;

        Node(const EOT& eo) : element(eo), discretized(Traits::nObjectives()) {}
    };

    typedef std::list<Node> archive_t;

    archive_t archive;
    std::vector<double> eps;
    
    public:
    
    static double direction(unsigned i) { return 1; }
    static double tol() { return 1e-6; }

            
    eoEpsilonArchive(const std::vector<double>& eps_) : eps(eps_) {
        if (eps.size() != Traits::nObjectives()) throw std::logic_error("eoEpsilonArchive: need one epsilon for each objective");    
    }

    bool empty() { return archive.size() == 0; }

    void operator()(const EOT& eo) {
        
        using std::cout;
        using std::endl;
        
        // discretize
        Node node(eo);
        for (unsigned i = 0; i < eo.fitness().size(); ++i) {
            double val = Traits::direction(i) * eo.fitness()[i];
            
            node.discretized[i] = floor(val/eps[i]);
        }

        using namespace dominance;

        typename archive_t::iterator boxIt = archive.end();
        // update archive
        
        for (typename archive_t::iterator it = archive.begin(); it != archive.end(); ++it) {
            dominance_result result = check<eoEpsilonArchive<EOT> >(node.discretized, it->discretized);

            switch (result) {
                case first : {                          // remove dominated archive member
                    it = archive.erase(it);
                    break;
                }
                case second : {
                    //cout << it->discretized[0] << ' ' << it->discretized[1] << " dominates " << node.discretized[0] << ' ' << node.discretized[1] << endl;
                    return;                   // new one does not belong in archive
                }
                case non_dominated : break;             // both non-dominated, put in different boxes
                case non_dominated_equal : {            // in same box
                    boxIt = it;
                    goto exitLoop; // break            
                }
            }

        }

exitLoop:
        // insert 
        if (boxIt == archive.end()) { // non-dominated, new box
            archive.push_back(node);
        } else { // fight it out
            int dom = node.element.fitness().check_dominance( boxIt->element.fitness() );
            
            switch (dom) {
                case 1: *boxIt = node; break;
                case -1: break;
                case 0: {
                    double d1 = 0.0;
                    double d2 = 0.0;
                    
                    for (unsigned i = 0; i < node.element.fitness().size(); ++i) {
                        double a = Traits::direction(i) * node.element.fitness()[i];
                        double b = Traits::direction(i) * boxIt->element.fitness()[i];

                        d1 += pow( (a - node.discretized[i]) / eps[i], 2.0);
                        d2 += pow( (b - node.discretized[i]) / eps[i], 2.0);
                    }
                    
                    if (d1 > d2) {
                        *boxIt = node; 
                    }
                    break;
                }
            }
        }
    }

    void appendTo(eoPop<EOT>& pop) const {
        for (typename archive_t::const_iterator it = archive.begin(); it != archive.end(); ++it) {
            pop.push_back( it->element );
        }
    }

    const EOT& selectRandom() const {
        int i = rng.random(archive.size());
        typename archive_t::const_iterator it = archive.begin();
        while (i-- > 0) it++;
        return it->element;
    }

};


#endif

