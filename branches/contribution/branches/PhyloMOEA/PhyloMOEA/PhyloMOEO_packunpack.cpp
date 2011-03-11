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
}

void unpack( PhyloMOEO &ind )
{
	//cout << "unpacking individual" << endl;
	string newickstring;
	::unpack(newickstring);
	//cout << newickstring << endl;
	ind.set_tree_template( *templatetree_ptr);
	phylotreeIND &tree = ind.get_tree();
	//reverse string 
	tree.read_newick2( newickstring );
	ind.invalidate();
}