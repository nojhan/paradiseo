void load(char* _fileName){
    FILE *f;
    int i,j;
    //open the file in read mode
    f=fopen(_fileName,"r" );
    //Verify if the file was open successfely
    if (f != NULL)
      fscanf(f,"%d",&n);
    else 
      printf("Le Fichier est vide\n");
   a=new int[n*n];
   b=new int[n*n];
    for (i=0;i<n;i++)
      for(j=0;j<n;j++)
	fscanf(f,"%d",&a[i*n+j]);      
    for (i=0;i<n;i++)
      for(j=0;j<n;j++)
	fscanf(f,"%d",&b[i*n+j]);
  }

template<class EOT>
void create(EOT &_solution){
    int random, temp;
    for (int i=0; i< n; i++) 
      _solution[i]=i;
    // we want a random permutation so we shuffle
    for (int i = 0; i < n; i++){
      random = rand()%(n-i) + i;
      temp = _solution[i];
      _solution[i] = _solution[random];
      _solution[random] = temp;
    }
  }  
