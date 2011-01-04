/** @mainpage Welcome to Evolving Objects

@section shortcuts In one word

The best place to learn about the features and approaches of %EO with the help of examples is to look at
the <a href="../../tutorial/html/eoTutorial.html">tutorial</a>.

Once you have understand the @ref design of %EO, you could search for advanced features by browsing the <a
href="modules.html">modules</a> page.


@section intro Introduction

 %EO is a template-based, ANSI-C++ evolutionary computation library which helps you to write your own stochastic optimization algorithms insanely fast.

It contains classes for almost any kind of evolutionary computation you might come
up to - at least for the ones we could think of. It is component-based, so that
if you don't find the class you need in it, it is very easy to subclass existing
abstract or concrete classes.

Designing an algorithm with %EO consists in choosing what components you want to use for your specific needs, just as building a structure with Lego blocks.

If you have a classical problem for which available code exists (for example if you have a black-box problem with real-valued variables), you will just choose components to form an algorithm and connect it to your fitness function (which computes the quality of a given solution).

If your problem is a bit more exotic, you will have to code a class that represents how your individuals (a solution to your problem) are represented, and perhaps some variations operators, but most of the other operators (selection, replacement, stopping criteria, command-line interface, etc.) are already available in %EO.



@section design Overall Design

%EO is a framework. It is oriented toward facilitating the design of adhoc evolutionary algorithms.
It is not (at the moment) a complete library of algorithms ready to use on canonical problems.

If you have a well-known problem and want to solve it as soon as possible, try another software.
If you have a real problem and want to build the best evolutionary algorithm to solve it, you've made
the good choice.

Bascially, %EO manipulate "individuals" with a "fitness", that is objects
encoding a solution to a given optimization problem, associated with
the quality of this solution. The fitness is defined in the %EO class,
but the representation of a solution cannot be as generic. Thus, %EO
massively use templates, so that you will not be limited by interfaces
when using your own representation.

Once you have a representation, you will build your own evolutionary algorithm
by assembling @ref Operators in @ref Algorithms.
In %EO, most of the objects are functors, that is classes with an operator(), that you
can call just as if they were classical functions. For example, an algorithm is a
functor, that manipulate a population of individuals, it will be implemented as a functor,
with a member like: operator()(eoPop<EOT>). Once called on a given population, it will
search for the optimum of a given problem.

Generally, operators are instanciated once and then binded in an algorithm by reference.
Thus, you can easily build your own algorithm by trying several combination of operators.

For a more detailled introduction to the design of %EO you can look at the
slides from a talk at EA 2001 or at the corresponding
article in Lecture Notes In Computer Science, 2310, Selected Papers from the 5th European Conference on Artificial Evolution:
    - http://portal.acm.org/citation.cfm?id=727742
    - http://eodev.sourceforge.net/eo/doc/LeCreusot.pdf
    - http://eodev.sourceforge.net/eo/doc/EO_EA2001.pdf
*/
