#ifndef DISTMAT_H
#define DISTMAT_H
#include <cstring>
#include <vector>
#include <set>
#include <iostream>
#include <utility>
#include <utils/eoRNG.h>
class DistMat {
public:
	virtual void load(const std::string)=0;
	virtual unsigned int numberOfVehicle()=0;
	virtual unsigned int numberOfDepot()=0;
	virtual unsigned int numberOfRetailer()=0;
	virtual unsigned int numberOfPlant()=0;
	virtual bool isConstantDemand()=0;
	virtual bool isConstantCapacity()=0;
	virtual double demand(unsigned int)=0;
	virtual double availability(unsigned int)=0;

	virtual double distance(unsigned int,unsigned int)=0;
	virtual double depotDistance(unsigned int,unsigned int)=0;
	virtual double plantDepotDistance(unsigned int,unsigned int)=0;
	virtual double maxLoad()=0;
	virtual double maxDuration()=0;
	typedef std::pair<unsigned int,double> ty;
	struct compaPair{
		bool operator()( ty a, ty b)const{
			return a.second<b.second;
		}
	};



	virtual std::multiset<ty, compaPair> getOrder(unsigned int)=0;
	virtual std::multiset<ty, compaPair> nearDepot(unsigned int)=0;
	virtual std::multiset<ty, compaPair> getOrderPlant(unsigned int)=0;
	virtual double cbOfPlant(unsigned int)=0;
	virtual double ccOfPlant(unsigned int)=0;
};
#endif
