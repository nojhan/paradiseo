#include <iostream>
#include <eo>

int main(int ac, char** av)
{
    eoParser parser(ac, av);

    unsigned int alpha1 = parser.createParam(10, "alpha1", "Alpha parameter").value();
    unsigned int alpha2 = parser.createParam(10, "alpha2", "Alpha parameter").value();
    unsigned int alpha3 = parser.createParam(10, "alpha3", "Alpha parameter").value();
    unsigned int alpha4 = parser.createParam(10, "alpha4", "Alpha parameter").value();
    unsigned int alpha5 = parser.createParam(10, "alpha5", "Alpha parameter").value();
    unsigned int alpha6 = parser.createParam(10, "alpha6", "Alpha parameter").value();

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_help(parser);

    std::cout << "alpha1: " << alpha1 << std::endl;
    std::cout << "alpha2: " << alpha2 << std::endl;
    std::cout << "alpha3: " << alpha3 << std::endl;
    std::cout << "alpha4: " << alpha4 << std::endl;
    std::cout << "alpha5: " << alpha5 << std::endl;
    std::cout << "alpha6: " << alpha6 << std::endl;

    return 0;
}
