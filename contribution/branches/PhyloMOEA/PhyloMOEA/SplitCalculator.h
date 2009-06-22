#ifndef _SPLITCALCULATOR_H_
#define _SPLITCALCULATOR_H_

#include <phylotreeIND.h>

struct split_info
{
	int left, right, num_nodes;
	split_info() {};
	split_info(int n) : left(n), right(-1), num_nodes(0) {};
};


class SplitCalculator : public  TreeCalculator <split_info> 
{

private :

    struct split_info temp;

public :

    /// virtual dtor here so there is no need to define it in derived classes
    virtual ~SplitCalculator() {}

    /// The pure virtual function that needs to be implemented by the subclass
    struct split_info operator()(phylotreeIND &tree)
    {
	// hash table
	int n = tree.number_of_taxons();
	struct split_info	**hash_table;  // hash that points struct info
	struct split_info 	*interior_node_info = new struct split_info[n-1];

	int idx_interior = 0;
	node_map<struct split_info*> interior_node(tree.TREE, NULL);
	edge_map<struct split_info*> interior_edge(tree.TREE, NULL);
	// node mapes
	int *map_nodes;
	int node_count = 0;
	int good_edges = 0;

	
	// allocate memory
	hash_table = new struct split_info*[n];
	for(int i=0; i<n; i++)hash_table[i] = NULL;
	map_nodes = new int[n];

	// step 1
	// select last taxon as root
	node invalid_node;
	node root1 = tree.taxon_number( n-1);

	// step 2 and 3
	postorder_Iterator it = tree.postorder_begin( root1);

	int l, r;
	while( *it != root1 )
	{
		struct split_info *father_info = interior_node [ it.ancestor() ] ;
		struct split_info *current_info = interior_node [ *it ] ;
			
		//cout << " node " << *it << " ancestral" << it.ancestor() << endl;
		if( tree.istaxon(*it) )
		{
			// update the map
			map_nodes[ tree.taxon_id( *it) ] = r = node_count;
			
			// check if is the leftmost
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &interior_node_info[idx_interior];
				idx_interior++;
				//father_info.left_most = *it;
				father_info->left = node_count;
			}
			//else father_info.right = node_count;	
			node_count++;
			++it;
		}
		else
		{
			int idx;
			l = current_info->left;
			interior_edge[ it.branch() ] = current_info;
			
			if( father_info == NULL )
			{
				interior_node [ it.ancestor() ] = father_info = &interior_node_info[idx_interior];
				idx_interior++;
				father_info->left = current_info->left;
			}

			++it;
			if (tree.istaxon(*it) || *it==root1) idx = r;
			else idx = l;
			
			current_info->right = r;
			// fill hash table
			hash_table[ idx ] = current_info;
		}
	}
	delete [] interior_node_info;
	delete [] map_nodes;
	delete [] hash_table;

    }

};
#endif