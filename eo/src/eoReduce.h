// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoReduce.h
//   Base class for population-merging classes
// (c) GeNeura Team, 1998
/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

   Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef eoReduce_h
#define eoReduce_h

//-----------------------------------------------------------------------------

#include <iostream>

// EO includes
#include <eoPop.h>     // eoPop
#include <eoFunctor.h>  // eoReduce
#include <utils/selectors.h>
#include <utils/eoLogger.h>

/**
 * eoReduce: .reduce the new generation to the specified size
   At the moment, limited to truncation - with 2 different methods,
   one that sorts the whole population, and one that repeatidely kills
   the worst. Ideally, we should be able to choose at run-time!!!

   @ingroup Replacors
*/
template<class EOT> class eoReduce: public eoBF<eoPop<EOT>&, unsigned, void>
{};

/** truncation method using sort
   @ingroup Replacors
 */
template <class EOT> class eoTruncate : public eoReduce<EOT>
{
    void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
    {
        if (_newgen.size() == _newsize)
            return;
        if (_newgen.size() < _newsize)
          throw std::logic_error("eoTruncate: Cannot truncate to a larger size!\n");

        _newgen.sort();
        _newgen.resize(_newsize);
    }
};

/** random truncation
   @ingroup Replacors
 * */
template <class EOT> class eoRandomReduce : public eoReduce<EOT>
{
    void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
    {
        if (_newgen.size() == _newsize)
            return;
        if (_newgen.size() < _newsize)
          throw std::logic_error("eoRandomReduce: Cannot truncate to a larger size!\n");

        // shuffle the population, then trucate
        _newgen.shuffle();
        _newgen.resize(_newsize);
    }
};

/**
EP truncation method (some global stochastic tournament +  sort)
Softer selective pressure than pure truncate
   @ingroup Replacors
*/
template <class EOT> class eoEPReduce : public eoReduce<EOT>
{
public:
    typedef typename EOT::Fitness Fitness;

    eoEPReduce(unsigned _t_size  ):
	t_size(_t_size)
    {
	if (t_size < 2)
	    {
		eo::log << eo::warnings << "Warning: EP tournament size should be >= 2. Adjusted" << std::endl;
		t_size = 2;
	    }
    }

    /// helper struct for comparing on std::pairs
    // compares the scores
    // uses the fitness if scores are equals ????
    typedef std::pair<float, typename eoPop<EOT>::iterator>  EPpair;
    struct Cmp {
	bool operator()(const EPpair a, const EPpair b) const
	{
	    if (b.first == a.first)
		return  (*b.second < *a.second);
	    return b.first < a.first;
	}
    };


    void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
    {
	unsigned int presentSize = _newgen.size();

	if (presentSize == _newsize)
            return;
        if (presentSize < _newsize)
	    throw std::logic_error("eoTruncate: Cannot truncate to a larger size!\n");
        std::vector<EPpair> scores(presentSize);
        for (unsigned i=0; i<presentSize; i++)
	    {
		scores[i].second = _newgen.begin()+i;
		Fitness fit = _newgen[i].fitness();
		for (unsigned itourn = 0; itourn < t_size; ++itourn)
		    {
			const EOT & competitor = _newgen[rng.random(presentSize)];
			if (fit > competitor.fitness())
			    scores[i].first += 1;
			else if (fit == competitor.fitness())
			    scores[i].first += 0.5;
		    }
	    }

        // now we have the scores
        typename std::vector<EPpair>::iterator it = scores.begin() + _newsize;
        std::nth_element(scores.begin(), it, scores.end(), Cmp());
        // sort(scores.begin(), scores.end(), Cmp());
        unsigned j;
	//      std::cout << "Les scores apres tri\n";
	//      for (j=0; j<scores.size(); j++)
	//        {
	//          std::cout << scores[j].first << " " << *scores[j].second << std::endl;
	//        }

	tmPop.reserve(presentSize);
	tmPop.clear();

        for (j=0; j<_newsize; j++)
	    {
		tmPop.push_back(*scores[j].second);
	    }

        _newgen.swap(tmPop);

        // erase does not work, but I'm sure there is a way in STL to mark
        // and later delete all inside a std::vector ??????
        // this would avoid all copies here

	//      it = scores.begin() + _newsize;
	//      while (it < scores.end())
	//        _newgen.erase(it->second);
    }
private:
    unsigned t_size;
    eoPop<EOT> tmPop;
};

/** a truncate class that does not sort, but repeatidely kills the worse.
To be used in SSGA-like replacements (e.g. see eoSSGAWorseReplacement)
   @ingroup Replacors
*/
template <class EOT>
class eoLinearTruncate : public eoReduce<EOT>
{
  void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
  {
    unsigned oldSize = _newgen.size();
    if (oldSize == _newsize)
      return;
    if (oldSize < _newsize)
      throw std::logic_error("eoLinearTruncate: Cannot truncate to a larger size!\n");
    for (unsigned i=0; i<oldSize - _newsize; i++)
      {
        typename eoPop<EOT>::iterator it = _newgen.it_worse_element();
        _newgen.erase(it);
      }
  }
};

/** a truncate class based on a repeated deterministic (reverse!) tournament
To be used in SSGA-like replacements (e.g. see eoSSGADetTournamentReplacement)
   @ingroup Replacors
*/
template <class EOT>
class eoDetTournamentTruncate : public eoReduce<EOT>
{
public:
  eoDetTournamentTruncate(unsigned _t_size):
    t_size(_t_size)
  {
    if (t_size < 2)
      {
          eo::log << eo::warnings << "Warning, Size for eoDetTournamentTruncate adjusted to 2" << std::endl;
        t_size = 2;
      }
  }

  void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
  {
    unsigned oldSize = _newgen.size();
    if (_newsize == 0)
      {
        _newgen.resize(0);
        return;
      }
    if (oldSize == _newsize)
      return;
    if (oldSize < _newsize)
      throw std::logic_error("eoDetTournamentTruncate: Cannot truncate to a larger size!\n");


    // Now OK to erase some losers
    std::cout << "oldSize - _newsize: " << oldSize - _newsize << std::endl;
    for (unsigned i=0; i<oldSize - _newsize; i++)
      {
        //OLDCODE EOT & eo = inverse_deterministic_tournament<EOT>(_newgen, t_size);
        //OLDCODE _newgen.erase(&eo);

        // Jeroen Eggermont stdc++v3  patch
        // in the new code from stdc++v3 an iterator from a container<T> is no longer an pointer to T
        // Because eo already contained a fuction using eoPop<EOT>::iterator's we will use the following

        _newgen.erase( inverse_deterministic_tournament(_newgen.begin(), _newgen.end(), t_size) );

      }
  }
private:
  unsigned t_size;
};

/** a truncate class based on a repeated deterministic (reverse!) tournament
To be used in SSGA-like replacements (e.g. see eoSSGAStochTournamentReplacement)
   @ingroup Replacors
*/
template <class EOT>
class eoStochTournamentTruncate : public eoReduce<EOT>
{
public:
  eoStochTournamentTruncate(double _t_rate):
    t_rate(_t_rate)
  {
    if (t_rate <= 0.5)
      {
          eo::log << eo::warnings << "Warning, Rate for eoStochTournamentTruncate adjusted to 0.51" << std::endl;
        t_rate = 0.51;
      }
    if (t_rate > 1)
      {
          eo::log << eo::warnings << "Warning, Rate for eoStochTournamentTruncate adjusted to 1" << std::endl;
        t_rate = 1;
      }
  }

  void operator()(eoPop<EOT>& _newgen, unsigned _newsize)
  {
    unsigned oldSize = _newgen.size();
    if (_newsize == 0)
      {
        _newgen.resize(0);
        return;
      }
    if (oldSize == _newsize)
      return;
    if (oldSize < _newsize)
      throw std::logic_error("eoStochTournamentTruncate: Cannot truncate to a larger size!\n");
    // Now OK to erase some losers
    for (unsigned i=0; i<oldSize - _newsize; i++)
      {
        //OLDCODE EOT & eo = inverse_stochastic_tournament<EOT>(_newgen, t_rate);
        //OLDCODE _newgen.erase(&eo);

        // Jeroen Eggermont stdc++v3  patch
        // in the new code from stdc++v3 an iterator from a container<T> is no longer an pointer to T
        // Because eo already contained a fuction using eoPop<EOT>::iterator's we will use the following

        _newgen.erase( inverse_stochastic_tournament(_newgen.begin(), _newgen.end(), t_rate) );


      }
  }

private:
  double t_rate;
};

//-----------------------------------------------------------------------------

#endif
