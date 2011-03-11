#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

using namespace std;

int main(int argc, char* argv[])
{
  unsigned int i, j;
    assert(argc ==4);
    string s,sub;
    ifstream infile(argv[1], std::ios::in);
    double p_value = atof(argv[2]);
    FILE* fp = fopen(argv[3], "wb");
    int tab[8];
    double tmp;
    for(unsigned int w=0; w<8; w++)
      tab[w]=0;
    // Ouvre le fichier de données :
    i=0;
    j=0;
    if (infile.is_open()){
        double tmp=0.0;
        //lit le fichier ligne par ligne
        while (std::getline (infile, s)){
	  if (s.length()>0){
	    sub=s.substr(35);
	    std::cout << "atof: " << atof(sub.c_str()) << endl;
	    tmp=atof(sub.c_str());
	    std::cout << "tmp: " << tmp << endl;
	    if( tmp < p_value)
	      tab[i]++;
	    j++;
	    if(j>6){
	      i++;
	      j=0;
	    }
	    std::cout << i << " " << j << endl;
	  }
        }	
	std::cout << "hop" << endl;
        //ferme le fichier
        infile.close();
    }
    fprintf(fp, "OneOne: %d\n", tab[7]);
    fprintf(fp, "OneND: %d\n", tab[6]);
    fprintf(fp, "OneFirst: %d\n", tab[5]);
    fprintf(fp, "OneAll: %d\n", tab[4]);
    fprintf(fp, "AllOne: %d\n", tab[3]);
    fprintf(fp, "AllND: %d\n", tab[2]);
    fprintf(fp, "AllFirst: %d\n", tab[1]);
    fprintf(fp, "AllAll: %d\n", tab[0]);
    return 0;
    fclose(fp);
}
