//-----------------------------------------------------------------------------
// t-eobin.cpp
//-----------------------------------------------------------------------------

#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream
#include <eo>         // eoBin

//-----------------------------------------------------------------------------

typedef eoBin<float> Chrom;

//-----------------------------------------------------------------------------

main()
{
  const unsigned SIZE = 8;
  unsigned i, j;

  Chrom chrom(SIZE), chrom2;
  
  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = true;
  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = false;
  cout << "chrom:  " << chrom << endl;
  chrom[0] = chrom[SIZE - 1] = true;

  cout << "chrom.className() = " << chrom.className() << endl;
  
  cout << "chrom:  " << chrom << endl
       << "chrom2: " << chrom2 << endl;
  
  ostrstream os;
  os << chrom;
  istrstream is(os.str());
  is >> chrom2;
  
  cout << "chrom:  " << chrom << endl
       << "chrom2: " << chrom2 << endl;
  
  fill(chrom.begin(), chrom.end(), false);
  cout << "--------------------------------------------------"
       << endl << "eoMonOp's aplied to .......... " << chrom << endl;
  
  eoBinRandom<Chrom> random;
  random(chrom);
  cout << "after eoBinRandom ............ " << chrom << endl;

  eoBinBitFlip<Chrom> bitflip;
  bitflip(chrom);
  cout << "after eoBitFlip .............. " << chrom << endl;
  
  eoBinMutation<Chrom> mutation(0.5);
  mutation(chrom);
  cout << "after eoBinMutation(0.5) ..... " << chrom << endl;

  eoBinInversion<Chrom> inversion;
  inversion(chrom);
  cout << "after eoBinInversion ......... " << chrom << endl;

  eoBinNext<Chrom> next;
  next(chrom);
  cout << "after eoBinNext .............. " << chrom << endl;

  eoBinPrev<Chrom> prev;
  prev(chrom);
  cout << "after eoBinPrev .............. " << chrom << endl;

  fill(chrom.begin(), chrom.end(), false);
  fill(chrom2.begin(), chrom2.end(), true);
  cout << "--------------------------------------------------"
       << endl << "eoBinOp's aplied to ... " 
       << chrom << " " << chrom2 << endl;

  eoBinCrossover<Chrom> xover;
  fill(chrom.begin(), chrom.end(), false);
  fill(chrom2.begin(), chrom2.end(), true);
  xover(chrom, chrom2);
  cout << "eoBinCrossover ........ " << chrom << " " << chrom2 << endl; 

  for (i = 1; i < SIZE; i++)
    {
      eoBinNxOver<Chrom> nxover(i);
      fill(chrom.begin(), chrom.end(), false);
      fill(chrom2.begin(), chrom2.end(), true);
      nxover(chrom, chrom2);
      cout << "eoBinNxOver(" << i << ") ........ " 
	   << chrom << " " << chrom2 << endl; 
    }

  for (i = 1; i < SIZE / 2; i++)
    for (j = 1; j < SIZE / 2; j++)
      {
	eoBinGxOver<Chrom> gxover(i, j);
	fill(chrom.begin(), chrom.end(), false);
	fill(chrom2.begin(), chrom2.end(), true);
	gxover(chrom, chrom2);
	cout  << "eoBinGxOver(" << i << ", " << j << ") ..... " 
	      << chrom << " " << chrom2 << endl; 
      }
  
  for (float r = 0.1; r < 1.0; r += 0.1)
    {
      eoBinUxOver<Chrom> uxover(r);
      fill(chrom.begin(), chrom.end(), false);
      fill(chrom2.begin(), chrom2.end(), true);
      uxover(chrom, chrom2);
      cout << "eoBinUxOver(" << r << ") ...... " 
	   << chrom << " " << chrom2 << endl; 
    }

  return 0;
}

//-----------------------------------------------------------------------------
