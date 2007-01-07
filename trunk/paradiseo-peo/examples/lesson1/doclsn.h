//! \mainpage Creating a simple ParadisEO-PEO Evolutionary Algorithm
//!
//! \section structure Introduction
//!
//! One of the first steps in designing an evolutionary algorihtm using the ParadisEO-PEO framework 
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
//! For a complete reference on the TSP-related classes and definitions please refer to the files under the <b>paradiseo-peo/examples/shared</b>.
//! After the installation phase you should end up having an <b>tspExample</b> executable file in the <b>paradiseo-peo/examples/lesson1</b> directory.
//! We will discuss testing and launching aspects later in the tutorial.
//! 
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
//! Each of the bellow refered header files may be found in the <b>pardiseo-peo/examples/shared</b> directory.
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
//!		Depending on your prolem, you may specify quadruple operators (two input individuals, two output resulting individuals), i.e. crossover operators,
//!		binary operators (one input individual and one output resulting individual), i.e. mutation operators, or combination of both types. As for the
//!		evaluation function, the signature of the peoEA constructor requires specifying a peoTransform derived object as transformation operator.
//!
//!		The transform operators, crossover and mutation, for the herein presended example are defined in the <b>order_xover.h</b> and the <b>city_swap.h</b>
//!		header files, respectively.
//!	</li>
//!	<li><i><b>continuation criterion</b></i> - the evolutionary algorithm evolves in an iterative manner; a continuation criterion has to be specified.
//!		One of the most common and simplest options considers a maximum number of generations. It is your choice whether to use
//!		a predefined EO class for specifying the continuation criterion or using a custom defined class. In the later case you have to
//!		make sure that your class derives the eoContinue class.<br/>
//!	</li>
//!	<li><i><b>selection strategy</b></i> - at each iteration a set of individuals are selected for applying the transform operators, in order
//!		to obtain the offspring population. As the specified parameter has to be derived from the eoSelect it is your option of whehter using
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
//!	<table style="border:none; border-spacing:0px;text-align:left; vertical-align:top; font-size:8pt;" border="0">
//!	<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!	<tr><td>eoPop< EOT > population( POP_SIZE, popInitializer ); &nbsp;</td> <td>// creation of a population with POP_SIZE individuals - the popInitializer is a functor to be called for each individual</td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoGenContinue< EOT > eaCont( NUM_GEN ); &nbsp;</td> <td>// number of generations for the evolutionary algorithm</td></tr>
//!	<tr><td>eoCheckPoint< EOT > eaCheckpointContinue( eaCont ); &nbsp;</td> <td>// checkpoint incorporating the continuation criterion - startpoint for adding other checkpoint objects</td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>peoSeqPopEval< EOT > eaPopEval( evalFunction ); &nbsp;</td> <td>// sequential evaluation functor wrapper - evalFunction represents the actual evaluation functor </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoRankingSelect< EOT > selectionStrategy; &nbsp;</td> <td>// selection strategy for creating the offspring population - a simple ranking selection in this case </td></tr>
//!	<tr><td>eoSelectNumber< EOT > eaSelect( selectionStrategy, POP_SIZE ); &nbsp;</td> <td>// the number of individuals to be selected for creating the offspring population </td></tr>
//!	<tr><td>eoRankingSelect< EOT > selectionStrategy; &nbsp;</td> <td>// selection strategy for creating the offspring population - a simple ranking selection in this case </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoSGATransform< EOT > transform( crossover, CROSS_RATE, mutation, MUT_RATE ); &nbsp;</td> <td>// transformation operator - crossover and mutation operators with their associated probabilities </td></tr>
//!	<tr><td>peoSeqTransform< EOT > eaTransform( transform ); &nbsp;</td> <td>// ParadisEO specific sequential operator - a parallel version may be specified in the same manner </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>eoPlusReplacement< EOT > eaReplace; &nbsp;</td> <td>// replacement strategy - for integrating the offspring resulting individuals in the initial population </td></tr>
//!	<tr><td> &nbsp; </td> <td> &nbsp; </td></tr>
//!	<tr><td>peoEA< EOT > eaAlg( eaCheckpointContinue, eaPopEval, eaSelect, eaTransform, eaReplace ); &nbsp;</td> <td>// ParadisEO evolutionary algorithm integrating the above defined objects </td></tr>
//!	<tr><td>eaAlg( population ); &nbsp;</td> <td>// specifying the initial population for the algorithm </td></tr>
//!	<tr><td>... &nbsp;</td> <td> &nbsp; </td></tr>
//!	</table>