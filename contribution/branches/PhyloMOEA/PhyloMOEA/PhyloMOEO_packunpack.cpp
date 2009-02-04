//
// C++ Implementation: PhyloMOEO_packunpack
//
// Description: 
//
//
// Author:  <>, (C) 2009
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <PhyloMOEO_packunpack.h> 

void pack( PhyloMOEO & ind)
{
	//cout << "packing individual" << endl;
	phylotreeIND & tree = ind.get_tree();
	string s = tree.newick_traverse2( true, false);
	
	::pack(s);
	/*edge edgeaux = *( tree.taxon_number(0).inout_edges_begin() );
	node root_traverse = edgeaux.opposite( tree.taxon_number(0) );
	//tree.convert_graph_to_tree( root_traverse, NULL);
	postorder_Iterator it = tree.postorder_begin( root_traverse );
	postorder_Iterator it2 = tree.postorder_end( root_traverse );
	double *blens = new double[(2*tree.number_of_taxons() -3 )];
	int top = tree.TREE.number_of_edges() -1;
	// reverse order ..... for GTL internals
	for( ; it!=it2; top--)
	{
		if( *it != root_traverse )
			
			blens[top] = tree.get_branch_length( it.branch() );
		++it;
	}
	for(int i=0; i< (2*tree.number_of_taxons() -3 ) ; i++) ::pack(blens[i]);
	delete [] blens; 

	/*cout << "\nenviado " << endl;
	cout << tree.newick_traverse2(false,false) << endl;*/

	//cout << "packing finished..." << endl;
	//::pack(blens);
}

void unpack( PhyloMOEO &ind )
{
	//cout << "unpacking individual" << endl;
	string newickstring;
	::unpack(newickstring);
	//cout << newickstring << endl;
	ind.set_tree_template( *templatetree_ptr);
	phylotreeIND &tree = ind.get_tree();
	
	tree.read_newick2( newickstring );
	ind.invalidate();
	/*edge edgeaux = *( tree.taxon_number(0).inout_edges_begin() );
	node root_traverse = edgeaux.opposite( tree.taxon_number(0) );
	tree.convert_graph_to_tree( root_traverse, NULL);
	postorder_Iterator it = tree.postorder_begin( root_traverse );
	postorder_Iterator it2 = tree.postorder_end( root_traverse );
	//double blens[2*Tind->number_of_taxons() -3];

	//MPI_Unpack(buffer, bufsize, pos, blens, 2*Tind->number_of_taxons() -3, MPI_DOUBLE, com);
	// reverse order ..... for GTL internals
	while(it!=it2)
	{
		double blen;
		if( *it != root_traverse )
		{
			::unpack(blen);
			tree.set_branch_length( it.branch(), blen );
		}
		++it;
	} 
	/*cout << "\n unpacking finished..." << endl;
	cout << "recibido \n" << endl;
	cout << newickstring << endl;
	cout << tree.newick_traverse2(false,false) << endl; */
}