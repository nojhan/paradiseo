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

#include "../../comparator/moeoParetoObjectiveVectorComparator.h"
#include "moeoArchiveIndex.h"
template < class ObjectiveVector >
class QuadTreeNode{
public:


	QuadTreeNode(ObjectiveVector& _objVec):objVec(_objVec),subTree(){}

	QuadTreeNode(const QuadTreeNode& _source):objVec(_source.objVec),subTree(_source.subTree){}

	QuadTreeNode& operator=(const QuadTreeNode& _src){
		(*this).objVec=_src.objVec;
		(*this).subTree=subTree;
		(*this).inserted=_src.is_inserted();
		if(inserted) (*this).index=_src.get_index();
		return *this;
	}

	ObjectiveVector& getVec(){
		return objVec;
	}

	/**
	 * @param _kSuccesor the k_successor of _child regarding this Node
	 * @param _child the child to link at the index _kSuccessor
	 * @return true if _child is inserted, false if there is already a child for this index
	 */
	bool setChild(unsigned int _kSuccesor, QuadTreeNode<ObjectiveVector>* _child){
		bool res = false;
		if((*this).subTree[_kSuccesor] == NULL){
			res=true;
			(*this).subTree[_kSuccesor]= _child;
		}
		return res;
	}




	std::map<unsigned int, QuadTreeNode<ObjectiveVector>*>& getSubTree(){
		return (*this).subTree;
	}

	void set_index(int idx){
		inserted=true;
		index=idx;
	}
	unsigned int get_index(){
		if (!inserted) std::cerr<<"moeoQuadTree getting index of a non-inserted node"<<std::endl;
		return index;
	}
	bool is_inserted(){
		return inserted;
	}

private:
	ObjectiveVector objVec;
	std::map<unsigned int, QuadTreeNode<ObjectiveVector>*> subTree;
	unsigned int index;
	bool inserted;
};



template < class MOEOT >
class moeoQuadTree : public moeoArchiveIndex<MOEOT>  {

	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	typedef typename std::map<unsigned int, QuadTreeNode<ObjectiveVector>*>::iterator QuadTreeIterator;
	typedef typename moeoArchiveIndex<MOEOT>::modif modif;
public:
	moeoQuadTree():root(NULL),current_size(0){
		bound=pow(2,ObjectiveVector::nObjectives())-1;
		comparator=new moeoParetoObjectiveVectorComparator<ObjectiveVector>();
	}

	~moeoQuadTree(){
		delete(comparator);
	}

	/**
	 * insert a _moeot in the index if it can be inserted
	 * @param @_moeot the individual ton insert
	 * @param _insert not used, should be changed...
	 */
	std::pair<bool,std::vector<modif> > operator()(const MOEOT& _moeot, bool _insert=true){
		std::pair<bool,std::vector<modif> > res;
		insert(_moeot.objectiveVector(),res);
		if (res.first){
			current_size=current_size+1-res.second.size();
		}
		return res;
	};

	/**
	 * apply the modif
	 * @param _update the modif to apply (move only)
	 * @return false if no problem occured
	 **/
	bool update( modif& _update){
		QuadTreeNode<ObjectiveVector>* node=find_node(_update.itemObjective,getRoot());
		if (node==NULL) return true;
		node->set_index(_update.newIdx);
		return false;
	}

	/**
	 * @paramm _obj the Objective Vector to insert into the tree.
	 * @return true if it is inserted
	 */
	void insert(ObjectiveVector _obj, std::pair<bool, std::vector<modif> > &res){
		//create a new node
		QuadTreeNode<ObjectiveVector>* tmp = new QuadTreeNode<ObjectiveVector>(_obj);
		//if the tree is empty, we have a new root!
		if(isEmpty()){
			root=tmp;
			tmp->set_index(0);
			res.first=true;
		}
		//else try to insert the new node in the tree
		else{
			res.first = insert_aux(tmp, root, NULL, 0, res.second);
			if(res.first) tmp->set_index(size()-1);
		}
	}

	/**
	 * @param _newnode the node to insert
	 * @param _tmproot the temporary root
	 * @param _parent the parent of _tmproot
	 * @param _succ the index of _parent where the _tmproot is linked
	 * @return true if the _newnode is inserted
	 */
	bool insert_aux(QuadTreeNode<ObjectiveVector>* _newnode, QuadTreeNode<ObjectiveVector>* _tmproot, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ, std::vector<modif> &modifs){
		bool res=false;
		bool dominated=false;

		unsigned int succ=k_succ(_newnode->getVec(), _tmproot->getVec());
		if(succ==bound){
			//_newnode is dominated by _tmproot
			delete(_newnode);
		}
		else if(succ==0){
			//_newnode dominates _tmproot
			replace(_newnode, _tmproot, _parent, _succ, modifs);
			res=true;
		}
		else{
			//dominance test1 (test if _newnode is dominated by the childs of _tmproot)
			if(!(_tmproot->getSubTree().empty())){
				QuadTreeIterator it=_tmproot->getSubTree().begin();
				while(!dominated && (it != _tmproot->getSubTree().end())){
					if((*it).second != NULL){
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
				//dominance test2 (test if _newnode dominates the childs of _tmproot)
				QuadTreeIterator it=_tmproot->getSubTree().begin();
				while(it != _tmproot->getSubTree().end()){
					if((*it).second != NULL){
						if( (succ < (*it).first) && ((succ & (*it).first) == succ)){
							test2(_newnode, (*it).second, _tmproot, (*it).first, modifs);
						}
					}
					it++;
				}

				//insertion
				if(_tmproot->setChild(succ, _newnode)){
					//the child is inserted
					res=true;
				}
				else{
					//else if the child is not inserted, insert it in the subtree
					res=insert_aux(_newnode, _tmproot->getSubTree()[succ], _tmproot, succ , modifs);
				}
			}
		}
		return res;
	}

	/*
	 * @param _objVec1
	 * @param _objVec2
	 * @return the k-successor of _objVec1 with respect to _objVec2
	 */
	unsigned int k_succ(const ObjectiveVector& _objVec1, const ObjectiveVector& _objVec2){
		unsigned int res=0;
		if(!(*comparator)(_objVec2, _objVec1)){
			for(unsigned int i=0; i < ObjectiveVector::nObjectives(); i++){
				if( (ObjectiveVector::minimizing(i) && ((_objVec1[i] - _objVec2[i]) >= (-1.0 * 1e-6 ))) ||
					(ObjectiveVector::maximizing(i) && ((_objVec1[i] - _objVec2[i]) <= 1e-6 ))){
					res+=pow(2,ObjectiveVector::nObjectives()-i-1);
				}
			}
		}
		return res;
	}

	/*
	 * replace the root by a new one
	 * @param _newnode thee new root
	 * @param _tmproot the old root
	 * @param _parent the parent of _tmproot
	 * @param _succ the index of _parent where the _tmproot is linked
	 */
	void replace(QuadTreeNode<ObjectiveVector>* _newnode, QuadTreeNode<ObjectiveVector>* _tmproot, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ, std::vector<modif> & res){
		if(!(_tmproot->getSubTree().empty())){
			//reconsider each son of the old root
			QuadTreeIterator it;
			for(it=(_tmproot->getSubTree()).begin(); it != (_tmproot->getSubTree()).end(); it++){
				if((*it).second!=NULL){
					reconsider(_newnode, (*it).second, res);
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
		modif new_modif(_tmproot->getVec(),_tmproot->get_index());
		res.push_back(new_modif);
		delete(_tmproot);
	}

	/**
	 * @param _newroot the new root
	 * @param _child a node to reconsider regarding tthe _newroot
	 */
	void reconsider(QuadTreeNode<ObjectiveVector>* _newroot, QuadTreeNode<ObjectiveVector>* _child, std::vector<modif> & res){
		unsigned int succ;
		//reconsider all child of _child
		if(!(_child->getSubTree().empty())){
			QuadTreeIterator it;
			for(it=(_child->getSubTree()).begin(); it != (_child->getSubTree()).end(); it++){
				if((*it).second != NULL){
					QuadTreeNode<ObjectiveVector>* tmp=(*it).second;
					_child->getSubTree()[(*it).first]=NULL;
					reconsider(_newroot, tmp, res );
				}
			}
		}
		succ=k_succ(_child->getVec(),_newroot->getVec());
		//if _child is dominated by the newroot, delete it
		if(succ==bound){
			modif new_modif(_child->getVec(),_child->get_index());
			res.push_back(new_modif);
			delete(_child);
		}
		//else reinsert it in the tree rooted at _newroot
		else if(_newroot->getSubTree()[succ] != NULL){
			reinsert(_newroot->getSubTree()[succ],_child);
		}
		else{
			_newroot->setChild(succ, _child);
		}
	}

	/**
	 * reinsert _node2 into _node1
	 * @param _node1 first node
	 * @param _node2 second node
	 */
	void reinsert(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2){
		//first resinsert all child of the second node into node1
		if(_node1 != _node2){
			unsigned int succ;
			if(!(_node2->getSubTree().empty())){
				QuadTreeIterator it;
				for(it=(_node2->getSubTree()).begin(); it != (_node2->getSubTree()).end(); it++){
					if((*it).second != NULL){
						QuadTreeNode<ObjectiveVector>* tmp=(*it).second;
						_node2->getSubTree()[(*it).first]=NULL;
						reinsert(_node1, tmp);
					}
				}
			}
			//insert node2 into node1
			succ=k_succ(_node2->getVec(),_node1->getVec());
			if(_node1->getSubTree()[succ] != NULL){
				reinsert(_node1->getSubTree()[succ],_node2);
			}
			else{
				_node1->setChild(succ, _node2);
			}
		}
	}

	/**
	 * remove a node
	 * @param _node the node to remove
	 * @param _parent its parent
	 * @param _succ the index of _parent where the _node is linked
	 */
	void remove(QuadTreeNode<ObjectiveVector>* _node, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ, std::vector<modif> & res){
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
		modif new_modif(_node->getVec(),_node->get_index());
		res.push_back(new_modif);
		delete(_node);
	}

	/**
	 * test if _node1 is dominated by _node2 (and recursivly by its childs)
	 * @param _node1 first node
	 * @param _node2 second node
	 */
	bool test1(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2){
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
		return res;
	}

	/**
	 * test if _node1 dominates _node2 (and recursivly its childs)
	 * @param _node1 first node
	 * @param _node2 second node
	 */
	void test2(QuadTreeNode<ObjectiveVector>* _node1, QuadTreeNode<ObjectiveVector>* _node2, QuadTreeNode<ObjectiveVector>* _parent, unsigned int _succ, std::vector<modif> & res){

		unsigned int succ;
		succ=k_succ(_node1->getVec(), _node2->getVec());
		if(succ==0){
			remove(_node2, _parent, _succ, res);
			if(_parent->getSubTree()[_succ]!=NULL)
				test2(_node1, _parent->getSubTree()[_succ], _parent, _succ, res);
		}
		else{
			QuadTreeIterator it=_node2->getSubTree().begin();
			while(it != _node2->getSubTree().end()){
				if((*it).second!=NULL){
					if( (succ & (*it).first) == succ){
						test2(_node1, (*it).second, _node2, (*it).first, res);
					}
				}
				it++;
			}
		}
	}

	//************* A REVOIR ************
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
	//***********************************

	/**
	 * @return if the tree is empty or not
	 */
	bool isEmpty(){
		return root==NULL;
	}

	/**
	 * @return a pointer on the root of the tree
	 */
	QuadTreeNode<ObjectiveVector>* getRoot(){
		return root;
	}

	/**
	 * the number of individual currently indexed
	 * @return the tree size
	 */
	unsigned int size(){
		return current_size;
	}


private:

	/**
	 * to find a node from  his objectiveVector
	 * @param _obj the objective to find
	 * @param current the node in which we are looking (to be able to recurse)
	 * @return the node with obj as objectiveVector, or null if it's not found
	 */
	QuadTreeNode<ObjectiveVector>* find_node(ObjectiveVector &_obj, QuadTreeNode<ObjectiveVector>* current){
		if (current->getVec()==_obj) return current;
		else{
			int succ=k_succ(current->getVec(),_obj);
			if(current->getSubTree()[succ]!=NULL)
				return find_node(_obj,current->getSubTree()[succ]);
			else{
				return NULL;
			}

		}

	}

	//pointer on the root of the tree
	QuadTreeNode<ObjectiveVector>* root;

	//size max of an index
	unsigned int bound;

	//Pareto comparator
	moeoParetoObjectiveVectorComparator<ObjectiveVector>* comparator;

	//current tree size
	int current_size;

};



#endif /*MOEOQUADTREE_H_*/
