#ifndef BIVRP_H_
#define BIVRP_H_
#include <BEO.h>
#include <VRP2.h>
#include <PLA.h>

class biVRP:public BEO <VRP2,PLA>{
	public:
		biVRP():BEO<VRP2,PLA>(){}
		biVRP(VRP2 & vrp, PLA &pla):BEO<VRP2,PLA>(vrp,pla){}


};	
#endif
