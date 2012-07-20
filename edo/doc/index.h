/** @mainpage Welcome to Evolving Distribution Objects

@section shortcuts In one word

%EDO is an extension of %EO oriented toward Estimation-of-Distribution-like
Algorithms.

You can search for advanced features by browsing the <a
href="modules.html">modules</a> page.

@section intro Introduction

%EDO is an extension of %EO, that facilitate the design and implementation of
stochastic search metaheuristics. It is based on the assumption that those
algorithms are updating a probability distribution, that is used to generate
a sample (a population, in %EO) of solutions (individuals, in %EO).

Basically, EDO decompose the <em>variation</em> operators of %EO in a set of
sub-operators that are binded by a <em>distribution</em>. Thus, most of the
representation-independent operators of %EO can be used in %EDO algorithms.

Apart from choosing which distribution he want to use as a model, the user is
not supposed to directly manipulate it. Using the same approach than within %EO,
the user has just to indicate what he want to use, without having to bother how
he want to use it.

On the designer side, it is still possible to implement specific operators
without having to change other ones.

<img src="edo_design.png" />

The two main operators are the <em>Estimators</em>, that builds a given
distribution according to a population and the <em>Samplers</em> that builds a
population according to a distribution. There is also <em>Modifiers</em> that
are here to change arbitrarily the parameters of a distribution, if necessary.

<img src="edo_distrib.png" />

*/
