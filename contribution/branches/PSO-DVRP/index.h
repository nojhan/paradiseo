/** @mainpage Welcome to PSO-DVRP for PARADISEO

@section Introduction

we propose a Particle Swarm Optimization metaheuristic (PSO) for resolving the Dynamic Vehicle Routing Problem (DVRP). This problem is the dynamic and real-time version of the conventional Vehicle Routing Problem (VRP).
In the classical VRP, the planning of tours is accomplished before the working day, whereas in the dynamic version of this problem, the planning of tours is made in a dynamic or real-time way.
The purpose is to try to insert the new customer orders in the already planned tours, when the vehicles are already on routes, and in minimizing the tours cost.

This metaheuristic is based on the Particle Swarm Optimization metaheuritic (PSO). It constitutes a great bio-inspired model of self-organization. Our approach is adaptive in both the representation of particles, and  the step of their evaluation.
Furthermore, we handle the DVRP as a mixed problem. This problem is treated as an Open Vehicle Routing Problem (OVRP) at the beginning of the working day. It consists only in optimizing the segments of routes traveled by vehicles.
At the end of working day, the problem is treated as a conventional VRP. In this case, the optimization is operated on complete tours. This means that we consider the return to the depot of the whole vehicle fleet.

As perspective, we expect to parallelize it and scale it on grid. A possible schema of parallelization is the multi-swarm, where several swarms evolve (move) in parallel on different calculation nodes.


@section authors AUTHORS

<TABLE>
<TR>
  <TD>Dolphin project-team INRIA Futurs, 2008.</TD>
  <TD>Mostepha Redouane Khouadjia <mr.khouadjia@ed.univ-lille1.fr> </TD>
  </TR>
</TABLE>


@section LICENSE

 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.

 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr


@section Paradiseo Home Page

<A href=http://paradiseo.gforge.inria.fr>http://paradiseo.gforge.inria.fr</A>

*/
