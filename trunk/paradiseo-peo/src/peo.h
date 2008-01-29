/*
* <peo.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peo_h_
#define __peo_h_

#include <eo>
#include <mo>
#include <moeo>


//! \mainpage The ParadisEO-PEO Framework
//!
//! \section intro Introduction
//!
//! ParadisEO is a white-box object-oriented framework dedicated to the reusable design
//! of parallel and distributed metaheuristics (PDM). ParadisEO provides a broad range of features including evolutionary
//! algorithms (EA), local searches (LS), the most common parallel and distributed models and hybridization
//! mechanisms, etc. This high content and utility encourages its use at European level. ParadisEO is based on a
//! clear conceptual separation of the solution methods from the problems they are intended to solve. This separation
//! confers to the user a maximum code and design reuse. Furthermore, the fine-grained nature of the classes
//! provided by the framework allow a higher flexibility compared to other frameworks. ParadisEO is one of the rare
//! frameworks that provide the most common parallel and distributed models. Their implementation is portable on
//! distributed-memory machines as well as on shared-memory multiprocessors, as it uses standard libraries such as
//! MPI, PVM and PThreads. The models can be exploited in a transparent way, one has just to instantiate their associated
//! provided classes. Their experimentation on the radio network design real-world application demonstrate their
//! efficiency.
//!
//! In practice, combinatorial optimization problems are often NP-hard, CPU time-consuming,
//! and evolve over time. Unlike exact methods, metaheuristics allow to tackle large-size problems
//! instances by delivering satisfactory solutions in a reasonable time. Metaheuristics are
//! general-purpose heuristics that split in two categories: evolutionary algorithms (EA) and local
//! search methods (LS). These two families have complementary characteristics: EA allow
//! a better exploration of the search space, while LS have the power to intensify the search in
//! promising regions. Their hybridization allows to deliver robust and better solutions
//!
//! Although serial metaheuristics have a polynomial temporal complexity, they remain
//! unsatisfactory for industrial problems. Parallel and distributed computing is a powerful way
//! to deal with the performance issue of these problems. Numerous parallel and distributed
//! metaheuristics (PDM) and their implementations have been proposed, and are available on
//! theWeb. They can be reused and adapted to his/her own problems. However, the user has to
//! deeply examine the code and rewrite its problem-specific sections. The task is tedious, errorprone,
//! takes along time and makes harder the produced code maintenance. A better way to
//! reuse the code of existing PDM is the reuse through libraries. These are often
//! more reliable as they are more tested and documented. They allow a better maintainability
//! and efficiency. However, libraries do not allow the reuse of design.
//!
//! \section parallel_metaheuristics Parallel and distributed metaheuristics
//!
//! \subsection parallel_distributed Parallel distributed evolutionary algorithms
//!
//! Evolutionary Algorithms (EA) are based on the iterative improvement of a
//! population of solutions. At each step, individuals are selected, paired and recombined in order
//! to generate new solutions that replace other ones, and so on. As the algorithm converges,
//! the population is mainly composed of individuals well adapted to the "environment", for
//! instance the problem. The main features that characterize EA are the way the population is
//! initialized, the selection strategy (deterministic/stochastic) by fostering "good" solutions,
//! the replacement strategy that discards individuals, and the continuation/stopping criterion
//! to decide whether the evolution should go on or not.
//!
//! Basically, three major parallel and distributed models for EA can been distinguished:
//! the island (a)synchronous cooperative model, the parallel evaluation of the
//! population, and the distributed evaluation of a single solution.
//! <ul>
//! 	<li> <i>Island (a)synchronous cooperative model</i>. Different EA are simultaneously deployed to
//!	cooperate for computing better and robust solutions. They exchange in an asynchronous
//!	way genetic stuff to diversify the search. The objective is to allow to delay the global
//!	convergence, especially when theEAare heterogeneous regarding the variation operators.
//!	The migration of individuals follows a policy defined by few parameters: the migration
//!	decision criterion, the exchange topology, the number of emigrants, the emigrants selection
//!	policy, and the replacement/integration policy.</li>
//!
//!	<li> <i>Parallel evaluation of the population</i>. It is required as it is in general the most timeconsuming.
//!	The parallel evaluation follows the centralized model. The farmer applies
//!	the following operations: selection, transformation and replacement as they require a
//!	global management of the population. At each generation, it distributes the set of new
//!	solutions between differentworkers. These evaluate and return back the solutions and their
//!	quality values. An efficient execution is often obtained particularly when the evaluation
//!	of each solution is costly. The two main advantages of an asynchronous model over
//!	the synchronous model are: (1) the fault tolerance of the asynchronous model; (2) the
//!	robustness in case the fitness computation can take very different computation times (e.g.
//!	for nonlinear numerical optimization). Whereas some time-out detection can be used to
//!	address the former issue, the latter one can be partially overcome if the grain is set to very
//!	small values, as individuals will be sent out for evaluations upon request of the workers.</li>
//!
//!	<li> <i>Distributed evaluation of a single solution</i>. The quality of each solution is evaluated in
//!	a parallel centralized way. That model is particularly interesting when the evaluation
//!	function can be itself parallelized as it is CPU time-consuming and/or IO intensive. In
//!	that case, the function can be viewed as an aggregation of a certain number of partial
//!	functions. The partial functions could also be identical if for example the problem to deal
//!	with is a data mining one. The evaluation is thus data parallel and the accesses to data
//!	base are performed in parallel. Furthermore, a reduction operation is performed on the
//!	results returned by the partial functions. As a summary, for this model the user has to
//!	indicate a set of partial functions and an aggregation operator of these.</li>
//! </ul>
//!
//! \subsection parallel_ls Parallel distributed local searches
//!
//! \subsubsection local_searches Local searches
//!
//! All metaheuristics dedicated to the improvement of a single solution
//! are based on the concept of neighborhood. They start from a solution randomly generated or
//! obtained from another optimization algorithm, and update it, step by step, by replacing the
//! current solution by one of its neighboring candidates. Some criterion have been identified to
//! differentiate such searches: the heuristic internal memory, the choice of the initial solution,
//! the candidate solutions generator, and the selection strategy of candidate moves. Three main
//! algorithms of local search stand out: Hill Climbing (HC), Simulated
//! Annealing (SA) and Tabu Search (TS).
//!
//! \subsubsection parallel_local_searches Parallel local searches
//!
//! Two parallel distributed models are commonly used in the literature: the parallel distributed
//! exploration of neighboring candidate solutions model, and the multi-start model.
//! <ul>
//!	<li><i>Parallel exploration of neighboring candidates</i>. It is a low-level Farmer-Worker model
//!	that does not alter the behavior of the heuristic. A sequential search computes the same
//!	results slower.At the beginning of each iteration, the farmer duplicates the current solution
//!	between distributed nodes. Each one manages some candidates and the results are returned to the farmer.
//!	The model is efficient if the evaluation of a each solution is time-consuming and/or there are a great
//!	deal of candidate neighbors to evaluate. This is obviously not applicable to SA since only one candidate
//!	is evaluated at each iteration. Likewise, the efficiency of the model for HC is not always guaranteed as
//!	the number of neighboring solutions to process before finding one that improves the current objective function may
//!	be highly variable.</li>
//!
//!	<li> <i>Multi-start model</i>. It consists in simultaneously launching several local searches. They
//!	may be heterogeneous, but no information is exchanged between them. The resultswould
//!	be identical as if the algorithms were sequentially run.Very often deterministic algorithms
//!	differ by the supplied initial solution and/or some other parameters. This trivial model is
//!	convenient for low-speed networks of workstations.</li>
//! </ul>
//!
//! \section hybridization Hybridization
//!
//! Recently, hybrid metaheuristics have gained a considerable interest. For many
//! practical or academic optimization problems, the best found solutions are obtained by
//! hybrid algorithms. Combinations of different metaheuristics have provided very powerful
//! search methods. Two levels and two modes
//! of hybridization have been distinguished: Low and High levels, and Relay and Cooperative modes.
//! The low-level hybridization addresses the functional composition of a single optimization
//! method. A function of a given metaheuristic is replaced by another metaheuristic. On the
//! contrary, for high-level hybrid algorithms the different metaheuristics are self-containing,
//! meaning no direct relationship to their internal working is considered. On the other hand,
//! relay hybridization means a set of metaheuristics is applied in a pipeline way. The output
//! of a metaheuristic (except the last) is the input of the following one (except the first).
//! Conversely, co-evolutionist hybridization is a cooperative optimization model. Each metaheuristic
//! performs a search in a solution space, and exchange solutions with others.
//!
//! \section paradiseo_goals Paradiseo goals and architecture
//!
//! The "EO" part of ParadisEO means Evolving Objects. EO is a C++ LGPL open source
//! framework and includes a paradigm-free Evolutionary Computation library (EOlib)
//! dedicated to the flexible design of EA through evolving objects superseding the most common
//! dialects (Genetic Algorithms, Evolution Strategies, Evolutionary Programming and
//! Genetic Programming). Furthermore, EO integrates several services including visualization
//! facilities, on-line definition of parameters, application check-pointing, etc. ParadisEO is an
//! extended version of the EO framework. The extensions include local search methods, hybridization
//! mechanisms, parallelism and distribution mechanisms, and other features that
//! are not addressed in this paper such as multi-objective optimization and grid computing. In
//! the next sections, we present the motivations and goals of ParadisEO, its architecture and
//! some of its main implementation details and issues.
//!
//! \subsection motivation Motivations and goals
//!
//! A framework is normally intended to be exploited by as many users as possible. Therefore,
//! its exploitation could be successful only if some important user criteria are satisfied. The
//! following criteria are the major of them and constitute the main objectives of the ParadisEO
//! framework:
//!
//! <ul>
//!	<li><i>Maximum design and code reuse</i>. The framework must provide for the user a whole
//!	architecture design of his/her solution method. Moreover, the programmer may redo as
//!	little code as possible. This objective requires a clear and maximal conceptual separation
//!	between the solution methods and the problems to be solved, and thus a deep domain
//!	analysis. The user might therefore develop only the minimal problem-specific code.</li>
//!
//!     <li><i>Flexibility and adaptability</i>. It must be possible for the user to easily add new features/
//!     metaheuristics or change existing ones without implicating other components. Furthermore,
//!     as in practice existing problems evolve and new others arise these have to be
//!     tackled by specializing/adapting the framework components.</li>
//!
//!	<li><i>Utility</i>. The framework must allow the user to cover a broad range of metaheuristics,
//!	problems, parallel distributed models, hybridization mechanisms, etc.</li>
//!
//!	<li><i>Transparent and easy access to performance and robustness</i>. As the optimization applications
//!	are often time-consuming the performance issue is crucial. Parallelism and
//!	distribution are two important ways to achieve high performance execution. In order to
//!	facilitate its use it is implemented so that the user can deploy his/her parallel algorithms in
//!	a transparent manner. Moreover, the execution of the algorithms must be robust to guarantee
//!	the reliability and the quality of the results. The hybridization mechanism allows
//!	to obtain robust and better solutions.</li>
//!
//!     <li><i>Portability</i>. In order to satisfy a large number of users the framework must support
//!     different material architectures and their associated operating systems.</li>
//! </ul>
//!
//! \subsection architecture ParadisEO architecture
//!
//! The architecture of ParadisEO is multi-layer and modular allowing to achieve the objectives
//! quoted above. This allows particularly a high flexibility and adaptability, an
//! easier hybridization, and more code and design reuse. The architecture has three layers
//! identifying three major categories of classes: <i>Solvers</i>, <i>Runners</i> and <i>Helpers</i>.
//! <ul>
//!	<li><i>Helpers</i>. Helpers are low-level classes that perform specific actions related to the evolution
//!	or search process. They are split in two categories: <i>Evolutionary helpers (EH)</i>
//!	and <i>Local search helpers (LSH)</i>. EH include mainly the transformation, selection and
//!	replacement operations, the evaluation function and the stopping criterion. LSH can be
//!	generic such as the neighborhood explorer class, or specific to the local search metaheuristic
//!	like the tabu list manager class in the Tabu Search solution method. On the
//!	other hand, there are some special helpers dedicated to the management of parallel and
//!	distributed models 2 and 3, such as the communicators that embody the communication
//!	services.
//!
//!	Helpers cooperate between them and interact with the components of the upper layer
//!	i.e. the runners. The runners invoke the helpers through function parameters. Indeed,
//!	helpers have not their own data, but they work on the internal data of the runners.</li>
//!
//!	<li><i>Runners</i>. The Runners layer contains a set of classes that implement the metaheuristics
//!	themselves. They perform the run of the metaheuristics from the initial state or
//!	population to the final one. One can distinguish the <i>Evolutionary runners (ER)</i> such as
//!	genetic algorithms, evolution strategies, etc., and <i>Local search runners (LSR)</i> like tabu
//!	search, simulated annealing and hill climbing. Runners invoke the helpers to perform
//!	specific actions on their data. For instance, an ER may ask the fitness function evaluation
//!	helper to evaluate its population. An LSR asks the movement helper to perform
//!	a given movement on the current state. Furthermore, runners can be serial or parallel
//!	distributed.</li>
//!
//!	<li><i>Solvers</i>. Solvers are devoted to control the evolution process and/or the search. They
//!	generate the initial state (solution or population) and define the strategy for combining
//!	and sequencing different metaheuristics. Two types of solvers can be distinguished.
//!	<i>Single metaheuristic solvers (SMS)</i> and <i>Multiple metaheuristics solvers (MMS)</i>. SMSs
//!	are dedicated to the execution of only one metaheuristic.MMS are more complex as they
//!	control and sequence several metaheuristics that can be heterogeneous. Solvers interact with
//!	the user by getting the input data and delivering the output (best solution, statistics,
//!	etc).</li>
//! </ul>
//!
//! According to the generality of their embedded features, the classes of the architecture split
//! in two major categories: <i>Provided classes</i> and <i>Required classes</i>. Provided classes embody
//! the factored out part of the metaheuristics. They are generic, implemented in the framework,
//! and ensure the control at run time. Required classes are those that must be supplied by the
//! user. They encapsulate the problem-specific aspects of the application. These classes are
//! fixed but not implemented in ParadisEO. The programmer has the burden to develop them
//! using the OO specialization mechanism.
//!
//! \section tutorials ParadisEO-PEO Tutorials
//!
//! The basisc of the ParadisEO framework philosophy are exposed in a few simple tutorials:
//! <ul>
//!	<li>
//!		<a href="lesson1/html/main.html" style="text-decoration:none;"> creating a simple ParadisEO evolutionary algorithm</a>;
//!	</li>
//! </ul>
//! All the presented examples have as case study the traveling salesman problem (TSP). Different operators and auxiliary objects were designed,
//! standing as a <a href="lsnshared/html/index.html" target="new" style="text-decoration:none;">common shared source code base</a>. While not being
//! part of the ParadisEO-PEO framework, it may represent a startpoint for a better understanding of the presented tutorials.
//!
//! \section LICENCE
//!
//!
//!This software is governed by the CeCILL license under French law and
//!abiding by the rules of distribution of free software.  You can  use,
//!modify and/ or redistribute the software under the terms of the CeCILL
//!license as circulated by CEA, CNRS and INRIA at the following URL
//!"http://www.cecill.info".
//!
//!As a counterpart to the access to the source code and  rights to copy,
//!modify and redistribute granted by the license, users are provided only
//!with a limited warranty  and the software's author,  the holder of the
//!economic rights,  and the successive licensors  have only  limited liability.
//!
//!In this respect, the user's attention is drawn to the risks associated
//!with loading,  using,  modifying and/or developing or reproducing the
//!software by the user in light of its specific status of free software,
//!that may mean  that it is complicated to manipulate,  and  that  also
//!therefore means  that it is reserved for developers  and  experienced
//!professionals having in-depth computer knowledge. Users are therefore
//!encouraged to load and test the software's suitability as regards their
//!requirements in conditions enabling the security of their systems and/or
//!data to be ensured and,  more generally, to use and operate it in the
//!same conditions as regards security.
//!The fact that you are presently reading this means that you have had
//!knowledge of the CeCILL license and that you accept its terms.
//!
//!ParadisEO WebSite : http://paradiseo.gforge.inria.fr
//!Contact: paradiseo-help@lists.gforge.inria.fr

#include "core/peo_init.h"
#include "core/peo_run.h"
#include "core/peo_fin.h"

#include "core/messaging.h"
#include "core/eoPop_mesg.h"
#include "core/eoVector_mesg.h"

#include "peoWrapper.h"

/* <------- components for parallel algorithms -------> */
#include "peoTransform.h"
#include "peoEvalFunc.h"
#include "peoPopEval.h"
#include "peoMoeoPopEval.h"

/* Cooperative island model */
#include "core/ring_topo.h"
#include "peoData.h"
#include "peoSyncIslandMig.h"
#include "peoAsyncIslandMig.h"

/* Synchronous multi-start model */
#include "peoMultiStart.h"
/* <------- components for parallel algorithms -------> */

/* Parallel PSO */
#include "peoPSOSelect.h"
#include "peoWorstPositionReplacement.h"
#include "peoGlobalBestVelocity.h"

#endif
