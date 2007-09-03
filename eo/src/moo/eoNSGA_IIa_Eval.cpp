
#include <moo/eoNSGA_IIa_Eval.h>

namespace nsga2a {

double distance(const std::vector<double>& f1, const std::vector<double>& f2, const std::vector<double>& range) {
    double dist = 0;
    for (unsigned i = 0; i < f1.size(); ++i) {
        double d = (f1[i] - f2[i]);
        dist += d*d;
    }
    return dist;
}


unsigned assign_worths(std::vector<detail::FitnessInfo> front, unsigned rank, std::vector<double>& worths) {

unsigned nDim = front[0].fitness.size();

// find boundary points
std::vector<unsigned> processed(nDim);

for (unsigned i = 1; i < front.size(); ++i) {
    for (unsigned dim = 0; dim < nDim; ++dim) {
        if (front[i].fitness[dim] > front[processed[dim]].fitness[dim]) {
            processed[dim] = i;
        }
    }
}
   
// assign fitness to processed and store in boundaries
std::vector<detail::FitnessInfo> boundaries;
for (unsigned i = 0; i < processed.size(); ++i) {
    
    worths[ front[ processed[i] ].index] = rank;
    
    //cout << "Boundary " << i << ' ' << front[processed[i]].index << ' ' << parents[ front[ processed[i] ].index]->fitness() << endl;

    boundaries.push_back( front[ processed[i] ] );
}
rank--;

// calculate ranges
std::vector<double> mins(nDim, std::numeric_limits<double>::infinity());
for (unsigned dim = 0; dim < boundaries.size(); ++dim) {
    for (unsigned i = 0; i < boundaries.size(); ++i) {
        mins[dim] = std::min( mins[dim], boundaries[i].fitness[dim] );
    }
}

std::vector<double> range(nDim);
for (unsigned dim = 0; dim < nDim; ++dim) {
    range[dim] = boundaries[dim].fitness[dim] - mins[dim];
}


// clean up processed (make unique) 
sort(processed.begin(), processed.end()); // swap out last first
for (unsigned i = 1; i < processed.size(); ++i) {
    if (processed[i] == processed[i-1]) {
        std::swap(processed[i], processed.back());
        processed.pop_back();
        --i;
    }
}
// remove boundaries from front
while (processed.size()) {
    unsigned idx = processed.back();
    //std::cout << idx << ' ' ;
    processed.pop_back();
    std::swap(front.back(), front[idx]);
    front.pop_back();
}
//std::cout << std::endl;

// calculate distances
std::vector<double> distances(front.size(), std::numeric_limits<double>::infinity());

unsigned selected = 0;
// select based on maximum distance to nearest processed point
for (unsigned i = 0; i < front.size(); ++i) {
    
    for (unsigned k = 0; k < boundaries.size(); ++k) {
        double d = distance( front[i].fitness, boundaries[k].fitness, range );
        if (d < distances[i]) {
            distances[i] = d;
        }
    }
    
    if (distances[i] > distances[selected]) {
        selected = i;
    }
}

 
while (!front.empty()) {
    
    detail::FitnessInfo last = front[selected];
    
    std::swap(front[selected], front.back());
    front.pop_back();

    std::swap(distances[selected], distances.back());
    distances.pop_back();

    // set worth
    worths[last.index] = rank--;

    if (front.empty()) break;

    selected = 0;

    for (unsigned i = 0; i < front.size(); ++i) {
        double d = distance(front[i].fitness, last.fitness, range);
        
        if (d < distances[i]) {
            distances[i] = d;
        }

        if (distances[i] >= distances[selected]) {
            selected = i;
        }
    }
}

return rank;
}

}
