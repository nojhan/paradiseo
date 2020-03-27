#include <iostream>
#include <string>

#include <eo>

struct OpInterface
{
    std::string _name;
    OpInterface(std::string name) : _name(name) {}
    virtual void operator()() = 0;
};

struct OpA : public OpInterface
{
    OpA(std::string name) : OpInterface(name) {}
    void operator()(){std::cout << _name << std::endl;}
};

struct OpB : public OpInterface
{
    OpB(std::string name, std::string suffix) : OpInterface(name+suffix) {}
    void operator()(){std::cout << _name << std::endl;}
};

int main(int /*argc*/, char** /*argv*/)
{
    // Forge container using names.
    eoForgeMap<OpInterface> named_factories;

    // Capture constructor's parameters and defer instanciation.
    named_factories.add<OpA>("OpA", "I'm A");
    named_factories.setup<OpA>("OpA", "I'm actually A"); // Edit
    named_factories.add<OpB>("OpB1", "I'm B", " prime");
    named_factories.add<OpB>("OpB2", "I'm a B", " junior");

    // Actually instanciante.
    OpInterface& opa  = named_factories.instanciate("OpA");
    OpInterface& opb1 = named_factories.instanciate("OpB1");

    // Call.
    opa();
    opb1();

    // Instanciate and call.
    named_factories.instanciate("OpB2")();


    // Forge container using indices.
    eoForgeVector<OpInterface> indexed_factories;

    // Capture constructor's parameters and defer instanciation.
    indexed_factories.add<OpA>("I'm A");
    indexed_factories.setup<OpA>(0, "I'm actually A"); // Edit
    indexed_factories.add<OpB>("I'm B", " prime");
    indexed_factories.add<OpB>("I'm a B", " junior");

    // Actually instanciante and call.
    indexed_factories.instanciate(0)();
    indexed_factories.instanciate(1)();
    indexed_factories.instanciate(2)();
}
