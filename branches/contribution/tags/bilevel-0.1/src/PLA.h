#ifndef PLA_H_
#define PLA_H_
#include <list>
#include <set>
#include <eo>
#include <EO.h>
#include <cmath>
class PLAObjectiveVectorTraits: public moeoObjectiveVectorTraits{
	public:
		static bool minimizing (int _i){
			return true;
		}
		static bool maximizing (int _i){
			return false;
		}
		static unsigned int nObjectives (){
			return 2;
		}

};

typedef moeoRealObjectiveVector<PLAObjectiveVectorTraits> PLAObjectiveVector;
class PLA: 
	public moeoVector<PLAObjectiveVector,double,double>
{
	public:
		PLA():plantNumber(0),depotNumber(0)
		{}
		PLA(std::vector<double> &_bui, DistMat &_mat):plantNumber(0),depotNumber(0)
		{
			init(_mat);
			for (int i=0;i<size();i++){
				operator[](i)=_bui[i];
			}
		}
		void init(DistMat & mat){
//			std::cout<<mat.numberOfVehicle()<<" "<<mat.numberOfRetailer()<<std::endl;
				resize(mat.numberOfPlant()*mat.numberOfDepot(),0);
				depotNumber=mat.numberOfDepot();
				plantNumber=mat.numberOfPlant();

		}

		void setTaux(int depot, int plant, double taux){
//			if (depot*plantNumber+plant>=size()) std::cout<<"set Taux"<<std::endl;
			operator[](depot*plantNumber+plant)=fabs(taux);
		}

		double getTaux(int depot, int plant)const{
//			if (depot*plantNumber+plant>=size()) std::cout<<"get Taux"<<std::endl;

			return fabs(operator[](depot*plantNumber+plant));
		}

		double sent(int depot, int plant, DistMat &mat)const {
			return getTaux(depot,plant)*mat.availability(plant);
		}

		void printOn(std::ostream &_os) const{
			double mmm=0;
			for(unsigned int i=0;i<size();i++){ 
				_os<<operator[](i)<<" ";
				mmm+=operator[](i);
			}
//			_os<<'\t'<<fitness()<<'\t'<<mmm;
		}
		//pas de surproduction
		void repairPlants(){
			for (unsigned int i=0;i<plantNumber;i++){
				for (unsigned int j=0;j<depotNumber;j++){
						setTaux(j,i,fabs(getTaux(j,i)));
				}
			}
			for (unsigned int i=0;i<plantNumber;i++){
				double sum=0;
				for (unsigned int j=0;j<depotNumber;j++){
					sum+=getTaux(j,i);
				}
				if (sum>1){
					double rap=1/sum;
					for (unsigned int j=0;j<depotNumber;j++){
						setTaux(j,i,getTaux(j,i)*rap);
					}
				}
			}

		}
		bool operator==(const PLA& _pla)const{
			double eps=0.000001;
			if (_pla.size()!=size()) return false;
			for (unsigned int i=0;i<size();i++){
				if (!(_pla[i]-operator[](i)<eps && _pla[i]-operator[](i)>-eps ))
					return false;
			}
			return true;
		}




	private:
		unsigned int plantNumber;
		unsigned int depotNumber;

};

#endif
