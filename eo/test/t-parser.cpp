// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
/* parser.cpp
 example of use of Parser.h

 (c) geneura team, 1999
-----------------------------------------------------------------------------*/

#include <eoParser.h>
#include <eoRNG.h>
#include <sys/time.h>

void GetOutputParam(Parser & parser, 
		    string & _string) {

  try {
    parser.AddTitle("Separate parameter: the output file name");
    _string = parser.getString("-O", "--OutputFile", "", "The output file name" );
  } catch (logic_error & e) {
    cout << e.what() << endl;
    parser.printHelp();
    exit(1);
  } catch (exception & e) {
    cout << e.what() << endl;
    exit(1);
  }
}

void sub(Parser & parser) {
  int i;
  cout << "Function sub:" << endl;

  try {
    parser.AddTitle("Private parameters of subroutine sub");
    i =  parser.getInt("-j", "--sint", "5", "private integer of subroutine" );
  } catch (logic_error & e) {
    cout << e.what() << endl;
    parser.printHelp();
    exit(1);
  } catch (exception & e) {
    cout << e.what() << endl;
    exit(1);
  }
  
  cout << "Read " << i << endl;
}


/// Uses the parser and returns param values
void getParams( Parser & parser, 
		unsigned & _integer,
		float & _floating,
		string & _string,
		vector<string> & _array,
		bool & _boolean) {
  
  try {
    _integer = parser.getInt("-i", "--int", "2", "interger number" );
    _floating = parser.getFloat("-f", "--float", "0.2", "floating point number" );
    _string = parser.getString("-s", "--string", "string", "a string" );
    _array = parser.getArray("-a", "--array", "a b", "an array enclosed within < >" );
    _boolean = parser.getBool("-b","--bool", "a bool value" );
  }
  catch (logic_error & e)
    {
      cout << e.what() << endl;
      parser.printHelp();
      exit(1);
    }
  catch (exception & e)
    {
      cout << e.what() << endl;
      exit(1);
    }

}

int main( int argc, char* argv[]) {
  
  unsigned in;
  float f;
  string s;
  vector<string> a;
  bool b;
 
  // Create the command-line parser
  Parser parser( argc, argv, "Parser example");
  InitRandom(parser);
  parser.AddTitle("General parameters");
  getParams(parser, in, f, s, a, b);

  cout << "\n integer: " << in << endl
       << " float: "<< f << endl
       << " string: /"<< s << "/" << endl
       << " boolean: "<< b << endl
       << " array:  < ";
  vector<string>::const_iterator i;
  for (i=a.begin() ; i<a.end() ; i++) {
    cout << *i << " ";
  }
  cout << ">" << endl << endl ;

  // call to the subroutine that also needs some parameters
  sub(parser);

  // writing all parameters
  // 
  // if programmer wishes, the name of the output file can be set as a parameter itself
  // otherwise it will be argv[0].status
  string OutputFileName;
  GetOutputParam(parser, OutputFileName);

  parser.outputParam(OutputFileName);
  if( parser.getBool("-h" , "--help" , "Shows this help")) {
    parser.printHelp();
    exit(1);
  }

  // but progrmamer should be careful to write the parser parameters
  // after the last bit that uses it has finished


  // Now the main body of the program

  for (int i=0; i<20; i++) {
    cout << rng.normal() << endl;
  }
  cout << "C'est fini" << endl;

  return 0;
}

