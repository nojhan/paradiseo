#include <moo/eoFrontSorter.h>

using namespace std;

namespace detail {

void one_objective(std::vector<FitnessInfo>& fitness, std::vector< std::vector<FitnessInfo> >& front, double tol)
{
    std::sort(fitness.begin(), fitness.end(), CompareOn(0, tol));
    
    front.clear(); 
    front.resize(1);
    front[0].push_back(fitness[0]);
    for (unsigned i = 1; i < fitness.size(); ++i) {
        if (fitness[i].fitness[0] < fitness[i-1].fitness[0]) { // keep clones in same front
            front.push_back( std::vector<FitnessInfo>() );
        }

        front.back().push_back( fitness[i] );
    }
}


/**
 * Optimization for two objectives. Makes the algorithm run in
 * complexity O(n log n) where n is the population size
 */


void two_objectives(std::vector<FitnessInfo>& fitness, std::vector< std::vector<FitnessInfo> >& front, double tol)
{
    std::sort(fitness.begin(), fitness.end(), CompareOn(0, tol));
    
    front.clear();
    
    std::vector<FitnessInfo> front_leader;

    for (unsigned i = 0; i < fitness.size(); ++i) {
        
        // find front through binary search
        vector<FitnessInfo>::iterator it = upper_bound( front_leader.begin(), front_leader.end(), fitness[i], CompareOn(1, tol));
        
        if (it == front_leader.end()) {
            front_leader.push_back(fitness[i]); 
            front.push_back( vector<FitnessInfo>(1, fitness[i]) );
        } else {
            *it = fitness[i]; // new 'best of front in second dimension'
            front[ it - front_leader.begin() ].push_back(fitness[i]); // add to front of nth dominated solutions
        }
    }
}

bool dominates(const FitnessInfo& a, const FitnessInfo& b, double tol) {
    bool better_on_one = false;
    
    for (unsigned i = 0; i < a.fitness.size(); ++i) {
        if (fabs(a.fitness[i] - b.fitness[i]) < tol) continue;

        if (a.fitness[i] < b.fitness[i]) return false; // worse on at least one other objective
        if (a.fitness[i] > b.fitness[i]) better_on_one = true;
    }

    return better_on_one;
}

void m_objectives(std::vector<FitnessInfo>& fitness, std::vector< std::vector<FitnessInfo> >& front, double tol) {
      unsigned i;

      std::vector<std::vector<unsigned> > S(fitness.size()); // which individuals does guy i dominate
      std::vector<unsigned> n(fitness.size()); // how many individuals dominate guy i

      unsigned j;
      for (i = 0; i < fitness.size(); ++i)
      {
        for (j = 0; j < fitness.size(); ++j)
        {
          if (  dominates(fitness[i], fitness[j], tol)  )
          { // i dominates j
            S[i].push_back(j); // add j to i's domination list
          }
          else if (dominates(fitness[j], fitness[i], tol))
          { // j dominates i, increment count for i, add i to the domination list of j
            n[i]++;
          }
        }
      }
    
      front.clear();
      front.resize(1);
      // get the first front out
      for (i = 0; i < fitness.size(); ++i)
      {
        if (n[i] == 0)
        {
          front.back().push_back( fitness[i] );
        }
      }
      
      while (!front.back().empty())
      { 
        front.push_back(vector<FitnessInfo>());
        vector<FitnessInfo>& last_front = front[front.size()-2];

        // Calculate which individuals are in the next front;

        for (i = 0; i < last_front.size(); ++i)
        {
          for (j = 0; j < S[ last_front[i].index ].size(); ++j)
          {
            unsigned dominated_individual = S[ last_front[i].index ][j];
            n[dominated_individual]--; // As we remove individual i -- being part of the current front -- it no longer dominates j

            if (n[dominated_individual] == 0) // it should be in the current front
            {
              front.back().push_back( fitness[dominated_individual] );
            }
          }
        }
      }
    
     front.pop_back(); // last front is empty;
}

void front_sorter_impl(std::vector<FitnessInfo>& fitness, std::vector< std::vector<FitnessInfo> >& front_indices, double tol) {
        
        switch (fitness[0].fitness.size())
	{
	    case 1:
		{
		    one_objective(fitness, front_indices, tol);
		    return;
		}
	    case 2:
		{
		    two_objectives(fitness, front_indices, tol);
		    return;
		}
	    default :
		{
		    m_objectives(fitness, front_indices, tol);
		}
	}
}

} // namespace detail

