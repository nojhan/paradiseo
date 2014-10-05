#include <iostream>
#include <vector>

#include "../../src/eo/utils/eoParserBoost.cpp"

int main(int ac, char** av)
{
    eoParserBoost parser(ac, av, "Test program.");

    unsigned int alpha1 = parser.createParam(10, "alpha1", "A default parameter", "a1", "Default values", false); // WARNING: "integer" is given and but type "unsigned int" is expected.
    int alpha2 = parser.createParam(10, "alpha2", "A default parameter", "a2", "Default values", false); // N.B. allows the same string shorthand ! But an exception will be thrown when using it...
    unsigned alpha3 = 10;
    parser.createParam(alpha3, "alpha3", "A default parameter", "c", "Default values", false); 
    unsigned int alpha4 = parser.createParam<unsigned>(10, "alpha4", "A implicit parameter", "d",  "Implicit values", true); 
    std::pair< unsigned, unsigned > alpha5 = parser.createParam(std::pair< unsigned, unsigned > (10, 20), "alpha5", "A implicit default parameter", "e",  "Implicit and default values"); // Must return the implicit and default values.
    parser.createParam(std::pair< std::string, std::string > ("", ""), "blank_option", "A implicit default parameter with empty values", "k", "Implicit and default options"); 

    // Requested value : if no value is given in command-line or by means of a configuration file, an exception will be thrown.
    parser.createParam<unsigned int>("alpha6", "A parameter requested value", "f", "Requested values");

    // Positional option  
    unsigned int alpha7 = parser.createParam((unsigned)10, "alpha7", "A positional default parameter", "g", "Positional options", false, 1); 
    //parser.createParam((unsigned)10, "alpha8", "A default parameter", "h", "Positional options", false, 1); // Get an exception: same position as the former parameter.
    parser.createParam((unsigned)5, "gui.alpha9", "A positional implicit parameter", "i", "Positional options", true, 3); 
    parser.createParam(std::pair< std::string, std::string > ("abc", "defg"), "My_string", "A positional implicit default parameter", "j", "Positional options", 17);
    
    std::cout << "alpha1: " << alpha1 << std::endl;
    std::cout << "alpha2: " << alpha2 << std::endl;
    std::cout << "alpha3: " << alpha3 << std::endl;
    std::cout << "alpha4: " << alpha4 << std::endl;
    std::cout << "alpha5 [default value]: " << alpha5.first << std::endl;
    std::cout << "alpha5 [implicit value]: " << alpha5.second << std::endl; 
    std::cout << "alpha7: " << alpha7 << std::endl;


    parser.processParams(true); // Set boolean to true in order to require the help configuration.

    bool userNeedsHelp = false;
    parser.getValue("help", userNeedsHelp);
    if (userNeedsHelp)
        exit(1); // Exit if help is requested. It has to be done here.

    alpha1 = 0; 
    try 
    {
        parser.getValue(std::string("alpha1"), alpha1); // Get an exception: the registered type (with createParam()) is different from the one which is given as a reference.
        std::cout << "alpha1: " << alpha1 << std::endl;
    }
    catch(std::exception& e) // Catch Boost exception.
    { 
        std::cout << "Catched exception: " << e.what() << std::endl; 
    }

    alpha2 = 0; 
    parser.getValue("alpha2", alpha2); // default value
    std::cout << "alpha2: " << alpha2 << std::endl; 

    unsigned int alpha6; 
    parser.getValue("alpha6", alpha6); // requested value
    std::cout << "alpha6: " << alpha6 << std::endl; 

    alpha4 = parser.getValue<unsigned int>("alpha4"); // implicit value
    std::cout << "alpha4: " << alpha4 << std::endl; 

    parser.writeSettings("o.txt");


    return 0;
}