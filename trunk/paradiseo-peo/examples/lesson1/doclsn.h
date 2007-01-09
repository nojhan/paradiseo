//! \mainpage Creating a simple ParadisEO-PEO Evolutionary Algorithm
//!
//! \section structure Introduction
//!
//! One of the first steps in designing an evolutionary algorithm using the ParadisEO-PEO framework 
//! consists in having a clear overview of the implemented algorithm. A brief pseudo-code description is offered 
//! bellow - the entire source code for the ParadisEO-PEO evolutionary algorithm is defined in the <b>peoEA.h</b>
//! header file. The main elements to be considered when building an evolutionary algorithm are the transformation
//! operators, i.e. crossover and mutation, the evaluation function, the continuation criterion and the selection
//! and replacement strategy.
//!
//!	<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!	<tr><td><b>do</b> { &nbsp;</td> <td> &nbsp; </td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; select( population, offsprings ); &nbsp;</td> <td>// select the offsprings from the current population</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; transform( offsprings ); &nbsp;</td> <td>// crossover and mutation operators are applied on the selected offsprings</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; evaluate( offsprings ); &nbsp;</td> <td>// evaluation step of the resulting offsprings</td></tr>
//!	<tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; replace( population, offsprings ); &nbsp;</td> <td>// replace the individuals in the current population whith individuals from the offspring population, according to a specified replacement strategy</td></tr>
//!	<tr><td>} <b>while</b> ( eaCheckpointContinue( population ) ); &nbsp;</td> <td>// checkpoint operators are applied on the current population</td></tr>
//!	</table>
//!
//! The peoEA class offers an elementary evolutionary algorithm implementation. The peoEA class has the underlying structure
//! for including parallel evaluation and parallel transformation operators, migration operators etc. Although there is 
//! no restriction on using the algorithms provided by the EO framework, no parallelism is provided - the EO implementation is exclusively sequential.
//! <br/>
//!
//! \section requirements Requirements
//!
//! You should have already installed the ParadisEO-PEO package - this requires several additional packages which should be already
//! included in the provided archive. The installation script has to be launched in order to configure and compile all the required
//! components. At the end of the installation phase you should end up having a directory tree resembling the following:
//! <b>
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; ...
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; paradiseo-mo
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; paradiseo-moeo
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; paradiseo-peo
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; docs
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; examples
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; lesson1
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; lesson2
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; shared
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; src
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...
//!	<br/>&nbsp;&nbsp;&nbsp;&nbsp; ...
//! </b>
//! <br/>
//!
//! The source-code for this tutorial may be found in the <b>paradiseo-peo/examples/lesson1</b> directory, in the <b>main.cpp</b> file.
//! We strongly encourage creating a backup copy of the file if you consider modifying the source code. For a complete reference on the 
//! TSP-related classes and definitions please refer to the files under the <b>paradiseo-peo/examples/shared</b>. After the installation
//! phase you should end up having an <b>tspExample</b> executable file in the <b>paradiseo-peo/examples/lesson1</b> directory.
//! We will discuss testing and launching aspects later in the tutorial.
//!
//! You are supposed to be familiar with working in C/C++ (with an extensive use of templates) and you should have at least an introductory
//! background in working with the EO framework.
//!
//! <hr/>
//! <b>NOTE</b>: All the presented examples have as case study the <i>Traveling Salesman Problem (TSP)</i>. All the presented tutorials rely
//! on a <a href="../../lsnshared/html/index.html" target="new">common shared source code</a> defining transformation operators, 
//! evaluation functions, etc. for the TSP problem. For a complete understanding of the presented tutorials please take your time for
//! consulting and for studying the additional underlying defined classes.
//! <hr/><br/>
//!
//! \section problemDef Problem Definition and Representation
//!
//! As we are not directly concerned with the <i>Traveling Salesman Problem</i>, and to some extent out of scope, no in depth details are offered
//! for the TSP. The problem requires finding the shortest path connecting a given set of cities, while visiting each of 
//! the specified cities only once and returning to the startpoint city. The problem is known to be NP-complete, i.e. no polynomial
//! time algorithm exists for solving the problem in exact manner.
//!
//! The construction of a ParadisEO-PEO evolutionary algorithm requires following a few simple steps - please take your time to study the signature
//! of the peoEA constructor:
//!
//!	<table border="0" width="100%">
//!	<tr><td style="vertical-align:top;">
//!		&nbsp;&nbsp;&nbsp;&nbsp; peoEA(
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; eoContinue< EOT >& __cont,
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; peoPopEval< EOT >& __pop_eval,
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; eoSelect< EOT >& __select,
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; peoTransform< EOT >& __trans,
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; eoReplacement< EOT >& __replace 
//!		<br/>&nbsp;&nbsp;&nbsp;&nbsp; );
//!	</td>
//!	<td style="vertical-align:top; text-align:center;">
//!		\image html peoEA.png
//!	</td></tr>
//!	</table>
//!
//! A few remarks have to be made: while most of the parameters are passed as EO-specific types, the evaluation and the transformation objects have to be
//! derived from the ParadisEO-PEO peoPopEval and peoTransform classes. Derived classes like the peoParaPopEval and peoParaSGATransform classes allow 
//! for parallel evaluation of the population and parallel transformation operators, respectively. Wrappers are provided thus allowing to make use 
//! of the EO classes.
//!
//! In the followings, the main required elements for building an evolutionary algorithm are enumerated. For complete details regarding the 
//! implementation aspects of each of the components, please refer to the <a href="../../lsnshared/html/index.html" target="new">common shared source code</a>.
//! Each of the bellow referred header files may be found in the <b>pardiseo-peo/examples/shared</b> directory.
//!
//! <ol>
//!	<li><i><b>representation</b></i> - the first decision to be taken concerns the representation of the individuals. You may create your
//!		own representation or you may use/derive one of the predefined classes of the EO framework. <br/>
//!
//!		For our case study, the TSP, each city is defined as a Node in the <b>node.h</b> header file - in fact an unsigned value defined
//!		as <b>typedef unsigned Node</b>. Moreover, each individual (of the evolutionary algorithm) is represented as a Route object, a vector of Node objects, in
//!		the <b>route.h</b> header file - <b>typedef eoVector< int, Node > Route</b>. The definition of the Route object implies two
//!		elements: (1) a route is a vector of nodes, and (2) the fitness is an integer value (please refer to the eoVector
//!		definition in the EO framework).
//!
//!		In addition you should also take a look in the <b>route_init.h</b> header file which includes the RouteInit class, defined for
//!		initializing in random manner Route objects.
//!	</li>
//!	<li><i><b>evaluation function</b></i> - having a representation model, an evaluation object has to be defined, implementing a specific
//!		fitness function. The designed class has to be derived (directly or indirectly) from the peoPopEval class - you have the choice of
//!		using peoSeqPopEval or peoParaPopEval for sequential and parallel evaluation, respectively. These classes act as wrappers requiring
//!		the specification of an EO evaluation object derived from the eoEvalFunc class - please refer to their respective documentation. <br/>
//!
//!		The fitness function for our TSP case study is implemented in the <b>route_eval.h</b> header file. The class is derived from the eoEvalFunc
//!		EO class, being defined as <b>class RouteEval : public eoEvalFunc< Route ></b>.
//!	</li>
//!	<li><i><b>transformation operators</b></i> - in order to assure the evolution of the initial population, transformation operators have to be defined.
//!		Depending on your problem, you may specify quadruple operators (two input individuals, two output resulting individuals), i.e. crossover operators,
//!		binary operators (one input individual and one output resulting individual), i.e. mutation operators, or combination of both types. As for the
//!		evaluation function, the signature of the peoEA constructor requires specifying a peoTransform derived object as transformation operator.
//!
//!		The transform operators, crossover and mutation, for the herein presented example are defined in the <b>order_xover.h</b> and the <b>city_swap.h</b>
//!		header files, respectively.
//!	</li>
//!	<li><i><b>continuation criterion</b></i> - the evolutionary algorithm evolves in an iterative manner; a continuation criterion has to be specified.
//!		One of the most common and simplest options considers a maximum number of generations. It is your choice whether to use
//!		a predefined EO class for specifying the continuation criterion or using a custom defined class. In the later case you have to
//!		make sure that your class derives the eoContinue class.<br/>
//!	</li>
//!	<li><i><b>selection strategy</b></i> - at each iteration a set of individuals are selected for applying the transform operators, in order
//!		to obtain the offspring population. As the specified parameter has to be derived from the eoSelect it is your option of whether using
//!		the EO provided selection strategies or implementing your own, as long as it inherits the eoSelect class.
//!
//!		For our example we chose to use the eoRankingSelect strategy, provided in the EO framework.
//!	</li>
//!	<li><i><b>replacement strategy</b></i> - once the offspring population is obtained, the offsprings have to be integrated back into the initial
//!		population, according to a given strategy. For custom defined strategies you have to inherit the eoReplacement EO class. We chose to
//!		use an eoPlusReplacement as strategy (please review the EO documentation for details on the different strategies available).
//!	</li>
//! </ol>
//! <br/>
//!
//! \section example A simple example for constructing a peoEA object
//!
//! The source code for this example may be found in the <b>main.cpp</b> file, under the <b>paradiseo-peo/examples/lesson1</b> directory. Please make sure you
//! At this point you have two options: (a) you can just follow the example without touching the <b>main.cpp</b> or, (b) you can start from scratch,
//! following the presented steps, in which case you are required make a backup copy of the <b>main.cpp</b> file and replace the original file with an 
//! empty one.
//!
//! <ol>
//!	<li> <b>include the necessary header files</b> - as we will be using Route objects, we have to include the files 
//!		which define the Route type, the initializing functor and the evaluation functions. Furthermore, in order to make use of 
//!		transform operators, we require having the headers which define the crossover and the mutation operators. 
//!		All these files may be found in the shared directory that we mentioned in the beginning. At this point you
//!		should have something like the following:<br/>
//!
//!		<pre>
//!		##include "route.h"
//!		##include "route_init.h"
//!		##include "route_eval.h"
//!		
//!		##include "order_xover.h"
//!		##include "city_swap.h"
//!		</pre>
//!		In addition we require having the <i>paradiseo</i> header file, in order to use the ParadisEO-PEO features, and a header specific
//!		for our problem, dealing with processing command-line parameters - the <b>param.h</b> header file. The complete picture at this point
//!		with all the required header files is as follows:<br/>
//!
//!		<pre>
//!		##include "route.h"
//!		##include "route_init.h"
//!		##include "route_eval.h"
//!		
//!		##include "order_xover.h"
//!		##include "city_swap.h"
//!
//!		##include "param.h"
//!
//!		##include &lt;paradiseo&gt;
//!		</pre>
//!		<b>NOTE</b>: the <b><i>paradiseo</i></b> header file is in fact a "super-header" - it includes all the esential ParadisEO-PEO header files. 
//!		It is at at your choice if you want use the <b><i>paradiseo</i></b> header file or to explicitly include different header files, 
//!		like the <b>peoEA.h</b> header file, for example.
//!		
//!	</li> 
//!	<li> <b>define problem specific parameters</b> - in our case we have to specify how many individuals we want to have in our population, the number
//!		of generations for the evolutionary algorithm to iterate and the probabilities associated with the crossover and mutation operators.<br/>
//!
//!		<pre>
//!		##include "route.h"
//!		##include "route_init.h"
//!		##include "route_eval.h"
//!		
//!		##include "order_xover.h"
//!		##include "city_swap.h"
//!
//!		##include "param.h"
//!
//!		##include &lt;paradiseo&gt;
//!
//!
//!		##define POP_SIZE 10
//!		##define NUM_GEN 100
//!		##define CROSS_RATE 1.0
//!		##define MUT_RATE 0.01
//!		</pre>
//!	</li>
//!	<li> <b>construct the skeleton of a simple ParadisEO-PEO program</b> - the main function including the code for initializing the ParadisEO-PEO
//!		environment, for loading problem data and for shutting down the ParadisEO-PEO environment. From this point on we will make 
//!		abstraction of the previous part referring only to the main function.<br/>
//!
//!		<pre>
//!		...
//!		
//!		int main( int __argc, char** __argv ) {
//!		
//!			<i>//</i> initializing the ParadisEO-PEO environment
//!			peo :: init( __argc, __argv );
//!		
//!			<i>//</i> processing the command line specified parameters
//!			loadParameters( __argc, __argv );
//!		
//!
//!			<i>//</i> EVOLUTIONARY ALGORITHM TO BE DEFINED
//!
//!		
//!			peo :: run( );
//!			peo :: finalize( );
//!			<i>//</i> shutting down the ParadisEO-PEO environment
//!		
//!			return 0;
//!		}
//!		</pre>
//!	</li>
//!	<li> <b>initialization functors, evaluation function and transform operators</b> - basically we only need to create instances for each of the
//!		enumerated objects, to be passed later as parameters for higher-level components of the evolutionary algorithm.<br/>
//!
//!		<pre>
//!		RouteInit route_init;	<i>//</i> random init object - creates random Route objects
//!		RouteEval full_eval;	<i>//</i> evaluator object - offers a fitness value for a specified Route object
//!
//!		OrderXover crossover;	<i>//</i> crossover operator - creates two offsprings out of two specified parents
//!		CitySwap mutation;	<i>//</i> mutation operator - randomly mutates one gene for a specified individual
//!		</pre>
//!	</li>
//!	<li> <b>construct the components of the evolutionary algorithm</b> - each of the components that has to be passed as parameter to the
//!		<b>peoEA</b> constructor has to be defined along with the associated parameters. Except for the requirement to provide the
//!		appropriate objects (for example, a peoPopEval derived object must be specified for the evaluation functor), there is no strict
//!		path to follow. It is your option what elements to mix, depending on your problem - we aimed for simplicity in our example.
//!
//!		<ul>
//!			<li> an initial population has to be specified; the constructor accepts the specification of an initializing object. Further,
//!				an evaluation object is required - the <b>peoEA</b> constructor requires a <b>peoPopEval</b> derived class.
//!			</li>
//!		</ul>
//!		<pre>
//!		eoPop< Route > population( POP_SIZE, route_init );	<i>//</i> initial population for the algorithm having POP_SIZE individuals
//!		peoSeqPopEval< Route > eaPopEval( full_eval );		// evaluator object - to be applied at each iteration on the entire population
//!		</pre>
//!		<ul>
//!			<li> the evolutionary algorithm continues to iterate till a continuation criterion is not met. For our case we consider
//!				a fixed number of generations. The continuation criterion has to be specified as a checkpoint object, thus requiring
//!				the creation of an <i>eoCheckPoint</i> object in addition.
//!			</li>
//!		</ul>
//!		<pre>
//!		eoGenContinue< Route > eaCont( NUM_GEN );		<i>//</i> continuation criterion - the algorithm will iterate for NUM_GEN generations
//!		eoCheckPoint< Route > eaCheckpointContinue( eaCont );	<i>//</i> checkpoint object - verify at each iteration if the continuation criterion is met
//!		</pre>
//!		<ul>
//!			<li> selection strategy - we are required to specify a selection strategy for extracting individuals out of the parent
//!				population; in addition the number of individuals to be selected has to be specified.
//!			</li>
//!		</ul>
//!		<pre>
//!		eoRankingSelect< Route > selectionStrategy;		<i>//</i> selection strategy - applied at each iteration for selecting parent individuals
//!		eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); <i>//</i> selection object - POP_SIZE individuals are selected at each iteration
//!		</pre>
//!		<ul>
//!			<li> transformation operators - we have to integrate the crossover and the mutation functors into an object which may be passed
//!				as a parameter when creating the <b>peoEA</b> object. The constructor of <b>peoEA</b> requires a <b>peoTransform</b> derived
//!				object. Associated probabilities have to be specified also.
//!			</li>
//!		</ul>
//!		<pre>
//!		<i>//</i> transform operator - includes the crossover and the mutation operators with a specified associated rate
//!		eoSGATransform< Route > transform( crossover, CROSS_RATE, mutation, MUT_RATE );
//!		peoSeqTransform< Route > eaTransform( transform );	<i>//</i> ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object
//!		</pre>
//!		<ul>
//!			<li> replacement strategy - required for defining the way for integrating the resulting offsprings into the initial population.
//!			At your option whether you would like to chose one of the predefined replacement strategies that come with the EO framework
//!			or if you want to define your own.
//!			</li>
//!		</ul>
//!		<pre>
//!		eoPlusReplacement< Route > eaReplace;			<i>//</i> replacement strategy - for replacing the initial population with offspring individuals
//!		</pre>
//!	</li>
//!	<li> <b>evolutionary algorithm</b> - having defined all the previous components, we are ready for instanciating an evolutionary algorithm.
//!		In the end we have to associate a population with the algorithm, which will serve as the initial population, to be iteratively evolved.
//!
//!		<pre>
//!		peoEA< Route > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace );
//!
//!		eaAlg( population );	// specifying the initial population for the algorithm, to be iteratively evolved
//!		</pre>
//!	</li>
//! </ol>
//!
//! If you have not missed any of the enumerated points, your program should be like the following:
//!
//! <pre>
//! ##include "route.h"
//! ##include "route_init.h"
//! ##include "route_eval.h"
//! 
//! ##include "order_xover.h"
//! ##include "city_swap.h"
//! 
//! ##include "param.h"
//! 
//! ##include <paradiseo>
//! 
//! 
//! ##define POP_SIZE 10
//! ##define NUM_GEN 100
//! ##define CROSS_RATE 1.0
//! ##define MUT_RATE 0.01
//! 
//! 
//! int main( int __argc, char** __argv ) {
//! 
//! 	<i>//</i> initializing the ParadisEO-PEO environment
//! 	peo :: init( __argc, __argv );
//! 
//! 
//! 	<i>//</i> processing the command line specified parameters
//! 	loadParameters( __argc, __argv );
//! 
//! 
//! 	<i>//</i> init, eval operators, EA operators -------------------------------------------------------------------------------------------------------------
//! 
//! 	RouteInit route_init;	<i>//</i> random init object - creates random Route objects
//! 	RouteEval full_eval;	<i>//</i> evaluator object - offers a fitness value for a specified Route object
//! 
//! 	OrderXover crossover;	<i>//</i> crossover operator - creates two offsprings out of two specified parents
//! 	CitySwap mutation;	<i>//</i> mutation operator - randomly mutates one gene for a specified individual
//! 	<i>//</i> ------------------------------------------------------------------------------------------------------------------------------------------------
//! 
//! 
//! 	<i>//</i> evolutionary algorithm components --------------------------------------------------------------------------------------------------------------
//! 
//! 	eoPop< Route > population( POP_SIZE, route_init );	<i>//</i> initial population for the algorithm having POP_SIZE individuals
//! 	peoSeqPopEval< Route > eaPopEval( full_eval );		<i>//</i> evaluator object - to be applied at each iteration on the entire population
//! 
//! 	eoGenContinue< Route > eaCont( NUM_GEN );		<i>//</i> continuation criterion - the algorithm will iterate for NUM_GEN generations
//! 	eoCheckPoint< Route > eaCheckpointContinue( eaCont );	<i>//</i> checkpoint object - verify at each iteration if the continuation criterion is met
//! 
//! 	eoRankingSelect< Route > selectionStrategy;		<i>//</i> selection strategy - applied at each iteration for selecting parent individuals
//! 	eoSelectNumber< Route > eaSelect( selectionStrategy, POP_SIZE ); <i>//</i> selection object - POP_SIZE individuals are selected at each iteration
//! 
//! 	<i>//</i> transform operator - includes the crossover and the mutation operators with a specified associated rate
//! 	eoSGATransform< Route > transform( crossover, CROSS_RATE, mutation, MUT_RATE );
//! 	peoSeqTransform< Route > eaTransform( transform );	<i>//</i> ParadisEO transform operator (please remark the peo prefix) - wraps an e EO transform object
//! 
//! 	eoPlusReplacement< Route > eaReplace;			<i>//</i> replacement strategy - for replacing the initial population with offspring individuals
//! 	<i>//</i> ------------------------------------------------------------------------------------------------------------------------------------------------
//! 
//! 
//! 	<i>//</i> ParadisEO-PEO evolutionary algorithm -----------------------------------------------------------------------------------------------------------
//! 
//! 	peoEA< Route > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace );
//! 	
//! 	eaAlg( population );	<i>//</i> specifying the initial population for the algorithm, to be iteratively evolved
//! 	<i>//</i> ------------------------------------------------------------------------------------------------------------------------------------------------
//! 
//! 
//! 	peo :: run( );
//! 	peo :: finalize( );
//! 	<i>//</i> shutting down the ParadisEO-PEO environment
//! 
//! 	return 0;
//! }
//! </pre>
//!
//!
//! \section testing Compilation and Execution
//!
//! First, please make sure that you followed all the previous steps in defining the evolutionary algorithm. Your file should be called <b>main.cpp</b> - please
//! make sure you do not rename the file (we will be using a pre-built makefile, thus you are required not to change the file names). Please make sure you 
//! are in the <b>paradiseo-peo/examples/lesson1</b> directory - you should open a console and you should change your current directory to the one of Lesson1.
//!
//! <b>Compilation</b>: being in the <b>paradiseo-peo/examples/lesson1</b> directory, you have to type <i>make</i>. As a result the <b>main.cpp</b> file
//! will be compiled and you should obtain an executable file called <b>tspExample</b>. If you have errors, please verify any of the followings:
//!
//!	<ul>
//!		<li>you are under the right directory - you can verify by typing the <i>pwd</i> command - you should have something like <b>.../paradiseo-peo/examples/lesson1</b></li>
//!		<li>you saved your modifications in a file called <b>main.cpp</b>, in the <b>paradiseo-peo/examples/lesson1</b> directory</li>
//!		<li>there are no differences between the example presented above and your file</li>
//!	</ul>
//!
//! <b>NOTE</b>: in order to successfully compile your program you should already have installed an MPI distribution in your system.
//!
//! <b>Execution</b>: the execution of a ParadisEO-PEO program requires having already created an environment for launching MPI programs. For <i>MPICH-2</i>,
//! for example, this requires starting a ring of daemons. The implementation that we provided as an example is sequential and includes no parallelism - we 
//! will see in the end how to include also parallelism. Executing a parallel program requires specifying a mapping of resources, in order to assing different
//! algorithms to different machines, define worker machines etc. This mapping is defined by an XML file called <b>schema.xml</b>, which, for our case, has
//! the following structure:
//!
//! <pre>
//!	<?xml version="1.0"?>
//!	
//!	<schema>
//!		<group scheduler="0">
//!			<node name="0" num_workers="0">
//!			</node>
//!			
//!			<node name="1" num_workers="0">
//!			<runner>1</runner>
//!			</node>
//!			
//!			<node name="2" num_workers="1">
//!			</node>
//!			<node name="3" num_workers="1">
//!			</node>
//!		</group>
//!	</schema>
//! </pre>
//!
//! Not going into details, the XML file presented above describes a mapping which includes four nodes, the first one having the role of scheduler,
//! the second one being the node on which the evolutionary algorithm is actually executed and the third and the fourth ones being slave nodes. Overall
//! the mapping says that we will be launching four processes, out of which only one will be executing the evolutionary algorithm. The other node entries
//! in the XML file have no real functionality as we have no parallelism in our program - the entries were created for you convenience, in order to provide
//! a smooth transition to creating a parallel program.
//!
//! Launching the program may be different, depending on your MPI distribution - for MPICH-2, in a console, in the <b>paradiseo-peo/examples/lesson1</b>
//! directory you have to type the following command:
//!
//!		<b>mpiexec -n 4 ./tspExample @lesson.param</b>
//!
//! <b>NOTE</b>: the "-n 4" indicates the number of processes to be launched. The last argument, "@lesson.param", indicates a file which specifies different
//! application specific parameters (the mapping file to be used, for example, whether to use logging or not, etc).
//!
//! The result of your execution should be similar to the following:
//!	<pre>
//! 	Loading '../data/eil101.tsp'.
//! 	NAME: eil101.
//! 	COMMENT: 101-city problem (Christofides/Eilon).
//! 	TYPE: TSP.
//! 	DIMENSION: 101.
//! 	EDGE_WEIGHT_TYPE: EUC_2D.
//! 	Loading '../data/eil101.tsp'.
//! 	NAME: eil101.
//! 	COMMENT: 101-city problem (Christofides/Eilon).
//! 	EOF.
//! 	TYPE: TSP.
//! 	DIMENSION: 101.
//! 	EDGE_WEIGHT_TYPE: EUC_2D.
//! 	EOF.
//! 	Loading '../data/eil101.tsp'.
//! 	NAME: eil101.
//! 	COMMENT: 101-city problem (Christofides/Eilon).
//! 	TYPE: TSP.
//! 	DIMENSION: 101.
//! 	EDGE_WEIGHT_TYPE: EUC_2D.
//! 	EOF.
//! 	Loading '../data/eil101.tsp'.
//! 	NAME: eil101.
//! 	COMMENT: 101-city problem (Christofides/Eilon).
//! 	TYPE: TSP.
//! 	DIMENSION: 101.
//! 	EDGE_WEIGHT_TYPE: EUC_2D.
//! 	EOF.
//! 	STOP in eoGenContinue: Reached maximum number of generations [100/100]
//!	</pre>
//!
//!
//! \section paraIntro Introducing parallelism
//!
//! Creating parallel programs with ParadisEO-PEO represents an easy task once you have the basic structure for your program. For experimentation,
//! in the <b>main.cpp</b> file, replace the line
//! <pre>
//!	peo<b>Seq</b>PopEval< Route > eaPopEval( full_eval );
//! </pre>
//! with
//! <pre>
//!	peo<b>Para</b>PopEval< Route > eaPopEval( full_eval );
//! </pre>
//! The second line only tells that we would like to evaluate individuals in parallel - this is very interesting if you have a time consuming fitness
//! evaluation function. If you take another look on the <b>schema.xml</b> XML file you will see the last two nodes being marked as slaves (the "num_workers"
//! attribute - these nodes will be used for computing the fitness of the individuals.
//!
//! At this point you only have to recompile your program and to launch it again - as we are not using a time consuming fitness fitness function, the
//! effects might not be visible - you may increase the number of individuals to experiment.
