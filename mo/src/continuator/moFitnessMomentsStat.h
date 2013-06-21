/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Lionel Parreaux <lionel.parreaux@gmail.com>

*/

#ifndef moFitnessMomentsStat_h
#define moFitnessMomentsStat_h

#include <utility>
#include <continuator/moStat.h>

/**
 * Statistic that saves the standard deviation of the fitness of the solutions during the search
 */
template <class EOT>
//class moFitnessMomentsStat : public moStat<EOT, std::pair<typename EOT::Fitness,typename EOT::Fitness> >
class moFitnessMomentsStat : public moStat<EOT, std::pair<double, double> >
{
public :
    typedef typename EOT::Fitness Fitness;
    //typedef std::pair<typename EOT::Fitness,typename EOT::Fitness> Pair;
    typedef std::pair<double, double> Pair;
    using moStat<EOT, Pair >::value;

    /**
     * Default Constructor
     * @param _reInitSol when true the best so far is reinitialized
     */
    moFitnessMomentsStat(bool _reInitSol = true)
    : moStat<EOT, Pair>(Pair(Fitness(), 0.0), "moments (average and stdev)"),
      reInitSol(_reInitSol), firstTime(true),
      nbSolutionsEncountered(0), currentAvg(0), currentVar(0)
    { }
    
    /**
     * Initialization of the best solution on the first one
     * @param _sol the first solution
     */
    virtual void init(EOT & _sol) {
    	if (reInitSol || firstTime)
    	{
    		value() = Pair(0.0,0.0);
    		nbSolutionsEncountered = currentAvg = currentVar = 0;
			firstTime = false;
    	}
    	/*else if (firstTime)
		{
			value() = 0.0;
			firstTime = false;
		}*/
    	operator()(_sol);
    }
    
	/**
	 * Update the best solution
	 * @param _sol the current solution
	 */
	virtual void operator()(EOT & _sol) {
		++nbSolutionsEncountered;
		double x = _sol.fitness();
		double oldAvg = currentAvg;
		currentAvg = oldAvg + (x - oldAvg)/nbSolutionsEncountered;
		if (nbSolutionsEncountered > 1) // <- not really necessary
		{
			//value() = (value()/nbSolutionsEncountered + _sol.fitness())/(nbSolutionsEncountered+1);
			double oldVar = currentVar;
			currentVar = oldVar + (x - oldAvg) * (x - currentAvg);
			value() = Pair(currentAvg, currentVar/nbSolutionsEncountered);
		}
	}

    /**
     * @return name of the class
     */
    virtual std::string className(void) const {
        return "moFitnessVarianceStat";
    }

protected:
    bool reInitSol;
    bool firstTime;
    double
      nbSolutionsEncountered
    , currentAvg
    , currentVar // actually the var * n
    ;

};

#endif // moFitnessMomentsStat_h
