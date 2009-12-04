/*
* <moeoQuadTree.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
* Jérémie Humeau
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
//-----------------------------------------------------------------------------

#ifndef MOEOQUADTREE_H_
#define MOEOQUADTREE_H_

#include <comparator/moeoParetoObjectiveVectorComparator.h>

template < class ObjectiveVector >
class QuadTreeNode{
public:
	QuadTreeNode(ObjectiveVector& _objVec):objVec(_objVec),subTree(){}

	QuadTreeNode(const QuadTreeNode& _source):objVec(_source.objVec),subTree(_source.subTree){}

	QuadTreeNode& operator=(const QuadTreeNode& _src){
		(*this).objVec=_src.objVec;
		(*this).subTree=subTree;
		return *this;
	}

	ObjectiveVector& getVec(){
		return objVec;
	}

	//return true if the child is inserted
	bool setChild(unsigned int _kSuccesor, QuadTreeNode<ObjectiveVector>* _child){
		std::cout << "enter setChild" << std::endl;
		bool res = false;
		if((*this).subTree[_kSuccesor] == NULL){
			res=true;
			(*this).subTree[_kSuccesor]= _child;
//			std::cout <<"setChild: " <<  getVec() << std::endl;
		}
		std::cout << "quit setChild" << std::endl;
		return res;
	}

	std::map<unsigned int, QuadTreeNode<ObjectiveVector>*>& getSubTree(){
		return (*this).subTree;
	}

private:
	ObjectiveVector objVec;
	std::map<unsigned int, QuadTreeNode<ObjectiveVector>*> subTree;
};



template < class ObjectiveVector >
class moeoQuadTree{

	typedef typename std::map<unsigned int, QuadTreeNode<ObjectiveVector>*>::iterator QuadTreeIterator;
public:
	moeoQuadTree():root(NULL){
		bound=pow(2,ObjectiveVector::nObjectives())-1;
		comparator=new moeoParetoObjectiveVectorComparator<ObjectiveVector>();
	}

	~moeoQuadTree(){
		delete(comparator);
	}

	bool insert(ObjectiveVector& _obj){
		bool res=false;
		QuadTreeNode<ObjectiveVector>* tmp = new QuadTreeNode<ObjectiveVector>(_obj);
		if(isEmpty()){
			root=tmp;
			res=true;
			std::cout << "insert case empty: " << root->getVec() << std::endl;
			std::cout << root << std::endl;
		}
		else{
			res = insert_aux(tmp, root, NULL, 0);
		}

		return res;
	}

	bool insert_aux(QuadTreeNode<ObjectiveVector>* _newnode, QuadTreeNode<ObjectiveVector>* _tmproot, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ){
		bool res=false;
		bool dominated=false;

		unsigned int succ=k_succ(_newnode->getVec(), _tmproot->getVec());
		if(succ==bound){
			//_newnode is dominated by _tmproot
			delete(_newnode);
		}
		else if(succ==0){
			//_newnode dominates _tmproot
			replace(_newnode, _tmproot, _parent, _succ);
			res=true;
		}
		else{
			//dominance test1
			if(!(_tmproot->getSubTree().empty())){
				QuadTreeIterator it=_tmproot->getSubTree().begin();
				while(!dominated && (it != _tmproot->getSubTree().end())){
					if((*it).second != NULL){
//						std::cout << "hop"<<std::endl;
//						std::cout << "first: " << (*it).first << ", bound: " << bound << ", xor: " << ((*it).first ^ bound) << std::endl;
						if( ((*it).first < succ) && (((succ ^ bound) & ((*it).first ^ bound)) == (succ ^ bound)) ){
							dominated = test1(_newnode, (*it).second);
						}
					}
					it++;
				}
			}
			if(dominated){
				//_newnode is dominated by a node of the subtree
				delete(_newnode);
			}
			else{
				//dominance test2
				QuadTreeIterator it=_tmproot->getSubTree().begin();
				while(it != _tmproot->getSubTree().end()){
					if((*it).second != NULL){
						if( (succ < (*it).first) && ((succ & (*it).first) == succ)){
							test2(_newnode, (*it).second, _tmproot, (*it).first);
						}
					}
					it++;
				}

				//insertion
				if(_tmproot->setChild(succ, _newnode)){
					//the child is inserted,
					res=true;
				}
				else{
					//else if the child is not inserted, insert it in the subtree
					res=insert_aux(_newnode, _tmproot->getSubTree()[succ], _tmproot, succ);
				}
			}
		}
		return res;

		//*******************************************************************
		//the tree is empty -> create a node and fix it at the root
//		if(isEmpty()){
//			root=tmp;
//			res=true;
//			std::cout << "insert case empty: " << root->getVec() << std::endl;
//			std::cout << root << std::endl;
//		}
//		else{
//			while(!stop){
//				//calulate the k-Successor de _obj wtih respect to the root
//				unsigned int succ=k_succ(_obj, root->getVec());
//				if(succ != bound){
//					if(succ == 0){
//						std::cout << "insert -> replace" << std::endl;
//						std::cout << root << std::endl;
//						replace(_obj);
//						realroot=root;
//						res=true;
//						stop=true;
//					}
//					else{
//						//dominance test1
//						if(!(root->getSubTree().empty())){
//							QuadTreeIterator it=root->getSubTree().begin();
//							while(!stop && (it != root->getSubTree().end())){
//								if((*it).second != NULL){
//									std::cout << "hop"<<std::endl;
//									std::cout << "first: " << (*it).first << ", bound: " << bound << ", xor: " << ((*it).first ^ bound) << std::endl;
//									if( ((*it).first < succ) && (((succ ^ bound) & ((*it).first ^ bound)) == (succ ^ bound)) ){
//
//										stop = test1(tmp, (*it).second);
//									}
//								}
//								it++;
//							}
//
//						}
//						if(!stop){
//							//dominance test2
//							QuadTreeIterator it=root->getSubTree().begin();
//							while(it != root->getSubTree().end()){
//								if((*it).second != NULL){
//									if( (succ < (*it).first) && ((succ & (*it).first) == succ)){
//										test2(tmp, (*it).second, root, (*it).first);
//									}
//								}
//								it++;
//							}
//							//insertion
//							QuadTreeNode<ObjectiveVector>* tmp = new QuadTreeNode<ObjectiveVector>(_obj);
//							std::cout << "insert case new son: " << root->getVec() << std::endl;
//							if(root->setChild(succ, tmp)){
//								std::cout << "\n\nthe root changed\n\n";
//								root=root->getSubTree()[succ];
//							}
//							else{
//								res=true;
//								stop=true;
//							}
//						}
//					}
//				}
//				else{
//					stop=true;
//				}
//			}
//			std::cout << "realroot: " << realroot->getVec() << std::endl;
//			root=realroot;
//		}
		//*******************************************************************
	}

	/*
	 * return the k-successor of _objVec1 with respect to _objVec2
	 * @param _objVec1
	 * @param _objVec2
	 */
	unsigned int k_succ(const ObjectiveVector& _objVec1, const ObjectiveVector& _objVec2){
		std::cout << "enter k_succ" << std::endl;
		unsigned int res=0;
		if(!(*comparator)(_objVec2, _objVec1)){
			for(int i=0; i < ObjectiveVector::nObjectives(); i++){
				if( (ObjectiveVector::minimizing(i) && ((_objVec1[i] - _objVec2[i]) >= (-1.0 * 1e-6 ))) ||
					(ObjectiveVector::maximizing(i) && ((_objVec1[i] - _objVec2[i]) <= 1e-6 ))){
					res+=pow(2,ObjectiveVector::nObjectives()-i-1);
				}
	//			if( (ObjectiveVector::minimizing(i) && (_objVec1[i] >= _objVec2[i])) ||
	//				(ObjectiveVector::maximizing(i) && (_objVec1[i] <= _objVec2[i]))){
	//				res+=pow(2,ObjectiveVector::nObjectives()-i-1);
	//			}
			}
		}
		std::cout << "quit k_succ" << std::endl;
		return res;
	}

	/*
	 * replace the old root by the new one
	 * @param _newroot
	 */
	void replace(QuadTreeNode<ObjectiveVector>* _newnode, QuadTreeNode<ObjectiveVector>* _tmproot, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ){
		std::cout << "enter replace: " << std::endl;

		if(!(_tmproot->getSubTree().empty())){
			//reconsider each son of the old root
			QuadTreeIterator it;
			for(it=(_tmproot->getSubTree()).begin(); it != (_tmproot->getSubTree()).end(); it++){
//				std::cout << "on passe ici" << std::endl;
				if((*it).second!=NULL){
//					std::cout << "replace: " << (*it).second->getVec() << std::endl;
					reconsider(_newnode, (*it).second);
//					std::cout << "end replacement" << std::endl;
				}
			}
		}
		//replace the old root by the new one
		if(_parent==NULL){
			root=_newnode;
		}
		else{
			_parent->getSubTree()[_succ]=_newnode;
		}
		//kill the old root
		delete(_tmproot);
//
//		QuadTreeNode<ObjectiveVector>* newroot = new QuadTreeNode<ObjectiveVector>(_newroot);
//
//		if(!(root->getSubTree().empty())){
//			QuadTreeIterator it;
//			for(it=(root->getSubTree()).begin(); it != (root->getSubTree()).end(); it++){
//				std::cout << "on passe ici" << std::endl;
//				if((*it).second!=NULL){
//					std::cout << "replace: " << (*it).second->getVec() << std::endl;
//					reconsider(newroot, (*it).second);
//					std::cout << "end replacement" << std::endl;
//				}
//			}
//		}
//		std::cout << "replace after reconsider" << std::endl;
//
//		delete(root);
//		root = newroot;
//		std::cout << root << " -> "<< root->getVec() << std::endl;
//		std::cout << "replace after change the root" << std::endl;
//		std::cout << "quit replace: " << std::endl;
	}

	void reconsider(QuadTreeNode<ObjectiveVector>* _newroot, QuadTreeNode<ObjectiveVector>* _child){
		std::cout << "enter reconsider: " << std::endl;
		unsigned int succ;
		if(!(_child->getSubTree().empty())){
			QuadTreeIterator it;
			for(it=(_child->getSubTree()).begin(); it != (_child->getSubTree()).end(); it++){
				if((*it).second != NULL){
					std::cout << "reconsider: " << (*it).second->getVec() << std::endl;
					QuadTreeNode<ObjectiveVector>* tmp=(*it).second;
					_child->getSubTree()[(*it).first]=NULL;

					reconsider(_newroot, tmp);
				}
			}
		}
		else{
			std::cout << "reconsider: no more child" << std::endl;
		}
		std::cout << "reconsider try to reinsert " << _child->getVec() << " in " << _newroot->getVec() << std::endl;
		succ=k_succ(_child->getVec(),_newroot->getVec());
		std::cout << "succ: " << succ << std::endl;
		if(succ==bound)
			delete(_child);
		else if(_newroot->getSubTree()[succ] != NULL){
//			std::cout << "hohoho" << std::endl;
			reinsert(_newroot->getSubTree()[succ],_child);
		}
		else{
//			std::cout << "houhouhou" << std::endl;
			_newroot->setChild(succ, _child);
		}
		std::cout << "quit reconsider: " << std::endl;
	}

	void reinsert(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2){
		std::cout << "enter reinsert: " << std::endl;
		if(_node1 != _node2){
			std::cout << "node1: " << _node1->getVec() << ", node2: " << _node2->getVec() << std::endl;
			unsigned int succ;
			if(!(_node2->getSubTree().empty())){
				QuadTreeIterator it;
				for(it=(_node2->getSubTree()).begin(); it != (_node2->getSubTree()).end(); it++){
					if((*it).second != NULL){
						std::cout << "reinsert: " << (*it).second->getVec() << std::endl;
						QuadTreeNode<ObjectiveVector>* tmp=(*it).second;
						_node2->getSubTree()[(*it).first]=NULL;
						reinsert(_node1, tmp);
					}
				}
			}
			succ=k_succ(_node2->getVec(),_node1->getVec());
			if(_node1->getSubTree()[succ] != NULL){
				reinsert(_node1->getSubTree()[succ],_node2);
			}
			else{
				_node1->setChild(succ, _node2);
			}
		}
		std::cout << "quit reinsert: " << std::endl;
	}

	void remove(QuadTreeNode<ObjectiveVector>* _node, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ){
		std::cout << "enter remove -> " << _node->getVec() << std::endl;
		printTree();
		unsigned int k=1;
		QuadTreeNode<ObjectiveVector>* tmp=NULL;
		_parent->getSubTree()[_succ]=NULL;
		while((k < (bound -1)) && _node->getSubTree()[k]==NULL){
			k++;
		}
		if(_node->getSubTree()[k]!=NULL){
			tmp =_node->getSubTree()[k];
			_parent->setChild(_succ, tmp);
		}
		k++;
		while(k < (bound -1)){
			if(_node->getSubTree()[k]!=NULL){
				reinsert(tmp ,_node->getSubTree()[k]);
			}
			k++;
		}
		delete(_node);
		std::cout << "quit remove: " << std::endl;
		printTree();
	}

	bool test1(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2){
		std::cout << "enter test1" << std::endl;
		bool res = false;
		unsigned int succ;
		succ=k_succ(_node1->getVec(), _node2->getVec());
		if(succ==bound){
			res=true;
		}
		else{
			QuadTreeIterator it=_node2->getSubTree().begin();
			while(!res && (it != _node2->getSubTree().end())){
				if((*it).second!=NULL){
					if( ((succ ^ bound) & ((*it).first ^ bound)) == (succ^bound)){
						res = res || test1(_node1, (*it).second);
					}
				}
				it++;
			}
		}
		std::cout << "quit test1" << std::endl;
		return res;
	}

	void test2(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ){
		std::cout << "enter test2" << std::endl;
//		printTree();
		unsigned int succ;
		succ=k_succ(_node1->getVec(), _node2->getVec());
		if(succ==0){
//			std::cout << "\n\n\nPEUT ETRE ICI\n\n\n";
			remove(_node2, _parent, _succ);
			if(_parent->getSubTree()[_succ]!=NULL)
				test2(_node1, _parent->getSubTree()[_succ], _parent, _succ);
		}
		else{
			QuadTreeIterator it=_node2->getSubTree().begin();
			while(it != _node2->getSubTree().end()){
				if((*it).second!=NULL){
					if( (succ & (*it).first) == succ){
						test2(_node1, (*it).second, _node2, (*it).first);
					}
				}
				it++;
			}
		}
		std::cout << "quit test2" << std::endl;
	}


	void printTree(){
		QuadTreeIterator it;
		if(!isEmpty()){
			std::cout << "root: " << root->getVec()  << " -> ";
			if(!(root->getSubTree().empty())){
				for(it=(root->getSubTree()).begin(); it != (root->getSubTree()).end(); it++){
					if((*it).second!=NULL)
						std::cout << (*it).second->getVec() << " ; ";
				}
				std::cout << std::endl;
				for(it=(root->getSubTree()).begin(); it != (root->getSubTree()).end(); it++){
					if((*it).second!=NULL){
						printChild((*it).second, (*it).first);
						std::cout << std::endl;
					}
				}
			}
		}
	}

	void printChild(QuadTreeNode<ObjectiveVector>* _child, unsigned int _key){
		QuadTreeIterator it;
		std::cout << "[" << _key << " : " << _child->getVec() << "] -> ";
		if(!(_child->getSubTree().empty())){
			for(it=(_child->getSubTree()).begin(); it != (_child->getSubTree()).end(); it++){
				if((*it).second!=NULL)
					std::cout << (*it).second->getVec() << " ; ";
			}
			std::cout << std::endl;
			for(it=(_child->getSubTree()).begin(); it != (_child->getSubTree()).end(); it++){
				if((*it).second!=NULL){
					printChild((*it).second, (*it).first);
					std::cout << std::endl;
				}
			}
		}
	}

	bool isEmpty(){
		return root==NULL;
	}

	QuadTreeNode<ObjectiveVector>* getRoot(){
		return root;
	}


private:


	QuadTreeNode<ObjectiveVector>* root;
	unsigned int bound;
	moeoParetoObjectiveVectorComparator<ObjectiveVector>* comparator;

};



#endif /*MOEOQUADTREE_H_*/
