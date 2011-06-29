#ifndef VRP_H_
#define VRP_H_
#include <list>
#include <set>
#include <eo>
#include <EO.h>
#include <DistMat.h>
#include <core/moeoObjectiveVectorTraits.h>
#include <core/moeoRealObjectiveVector.h>
#include <core/moeoVector.h>
class VRP2ObjectiveVectorTraits: public moeoObjectiveVectorTraits{
	public:
		static bool minimizing (int _i){
			return false;
		}
		static bool maximizing (int _i){
			return true;
		}
		static unsigned int nObjectives (){
			return 1;
		}

};

typedef moeoRealObjectiveVector<VRP2ObjectiveVectorTraits> VRP2ObjectiveVector;
class VRP2: 
	public moeoVector<VRP2ObjectiveVector,int,double>
{


	public:
		VRP2():retailerNumber(0),vehicleNumber(0),depotNumber(0),inited(false){}
		VRP2(std::vector<int> &_bui, DistMat& _mat){
			init(_mat);
			for (int i=0;i<size();i++){
				operator[](i)=_bui[i];
			}
		}
		void init(DistMat & mat){
			resize(mat.numberOfVehicle()+mat.numberOfRetailer()-1,-1);
			retailerNumber=mat.numberOfRetailer();
			vehicleNumber=mat.numberOfVehicle();
			depotNumber=mat.numberOfDepot();
			voituresIdx.resize(vehicleNumber-1);
			inited=true;

		}
		void printOn(std::ostream &_os) const{
			int sum=0;
			for(unsigned int i=0;i<size();i++){ 
				if (isVehicle(operator[](i))) _os<<"("<<operator[](i)<<")";
				else _os<<"["<< operator[](i)<<"]";
				sum+=operator[](i);
			}
			_os<<'\t'<<fitness()<<" MMM"<<sum;
		}
		bool isVehicle(unsigned int i) const{
			return i>=retailerNumber &&i<size();
		}
		bool isVehicleAt(int i) const {
			if (i>=size()) return false;

			return i==-1 || isVehicle(operator[](i));
		}
		int depotOfVehicle(int i) const {
			return (i+1)/(vehicleNumber/depotNumber);
		}
		int depotOfVehicleAt(int i) const{
			return depotOfVehicle(operator[](i)-retailerNumber);
		}

		double depotDemand(DistMat &_mat , const unsigned int i) {
			int firstvoiture=i*vehicleNumber/depotNumber-1;
			double res=0;
			for (unsigned int j=0;j<vehicleNumber/depotNumber;j++){
				res+=chargeOfVehicleAt(_mat,(firstvoiture==-1 && j==0)?-1:idxOfVoiture(firstvoiture+j));
			}
			return res;
		}

		double chargeOfVehicleAt(DistMat &_mat, int idx) {
			double res=0;
			unsigned int i=(idx==-1?0:idx+1);
			if (isVehicleAt(idx)|| idx==-1){
				while (!isVehicleAt(i)&& i<size()){
					res+=_mat.demand(operator[](i));
					i++;
				}
			}
			return res;
		}
		bool check()const {
			std::cout<<"check"<<std::endl;
			for (int i=0;i<size();i++){
				std::cout<<operator[](i)<<" ";
			}
			std::cout<<std::endl;
			for (int i=0;i<size();i++){
				if (std::count(begin(),end(),i)!=1){
					std::cout<<"PAS UNE PERMUT ("<<i<<")"<<std::endl;
					std::cout<<"taille: "<<size()<<std::endl;
					throw std::runtime_error("plus une permut");
					return false;
				}
			}
			std::cout<<"check ok"<<std::endl;
			return true;
		}

		unsigned int idxOfVoiture(int i){
			if (!checkIdx()){
				for (unsigned int i=0;i<size();i++){
					if (isVehicleAt(i)){
						voituresIdx[operator[](i)-retailerNumber]=i;
					}
				}
			}
			return voituresIdx[i];

		}

		bool operator==(const VRP2 &_vrp)const{
			if (_vrp.size()!=size()) return false;
			for (unsigned int i=0;i<_vrp.size();i++){
				if (_vrp[i]!=operator[](i)) return false;
			}
			return true;
		}

	private:
		unsigned int retailerNumber;
		unsigned int vehicleNumber;
		unsigned int depotNumber;
		std::vector<unsigned int> voituresIdx;

		bool checkIdx(){
			for (unsigned int i=0;i<voituresIdx.size();i++){
				unsigned int val=operator[](voituresIdx[i]) - retailerNumber;
				if (val!=i) return false;

			}
			return true;
		}
		bool inited;

};

#endif
