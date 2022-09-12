
#include <frictionless/moBinaryPartition.h>
#include <frictionless/moBinaryPartitionSwapNeighbor.h>
#include <frictionless/moBinaryPartitionSwapNeighborhood.h>

int main()
{
    using Signature = moBinaryPartition<double>;
    const size_t genes_nb = 4;

    /**********************************************
     * Test if neighborhood has all neighbors.
     **********************************************/
    Signature geneset(genes_nb);
    std::clog << "Available genes:";
    for(size_t g : geneset.rejected) {
        std::clog << " " << g;
    }
    std::clog << std::endl;

    const size_t n = 2;
    for(size_t i=0; i < n; ++i) {
        geneset.select(i);
    }

    std::clog << "Init geneset: " << geneset << std::endl;
    std::clog << std::endl;

    moBinaryPartitionSwapNeighborhood<Signature> neighborhood;

    // Save generated solutions for testing.
    std::vector<Signature> solutions;

    // Follows the framework's workflow (see moRandomBestHCexplorer):
    // 1) if hasNeighbor()
    // 2) neighborhood.init(…)
    // 3) [eval]
    // 4) while neighborhood.cont(…)
    // 5) neighborhood.next(…)
    // 6) [eval]
    // … loop.
    if(neighborhood.hasNeighbor(geneset)) {

        moBinaryPartitionSwapNeighbor<Signature> neighbor(n);
        neighborhood.init(geneset, neighbor);
        std::clog << "Init neighbor: " << neighbor << std::endl;

        // Print what it looks like.
        std::clog << "Current geneset: " << geneset << std::endl;
        Signature new_geneset = geneset;
        neighbor.move(new_geneset);
        std::clog << "Moved to solution: " << new_geneset << std::endl;
        solutions.push_back(new_geneset);
        std::clog << std::endl;

        while(neighborhood.cont(geneset)) {
            // Generate next neighbor.
            neighborhood.next(geneset, neighbor);
            std::clog << "New neighbor: " << neighbor << std::endl;

            // Print what it looks like.
            std::clog << "Current geneset: " << geneset << std::endl;
            Signature new_geneset = geneset;
            neighbor.move(new_geneset);
            std::clog << "Moved to solution: " << new_geneset << std::endl;
            solutions.push_back(new_geneset);

            // Double check that one can moveBack and get the same solution.
            neighbor.moveBack(new_geneset);
            assert(new_geneset == geneset);
            std::clog << std::endl;
        }
    }

    std::clog << "Generated " << solutions.size() << " neighbors of: " << geneset << std::endl;
    for(Signature s : solutions) {
        std::clog << "\t" << s << std::endl;
    }
    assert(solutions.size() == 4);

    /**********************************************
     * Test if a full solution does not have neighbor.
     **********************************************/
    Signature full(genes_nb);
    for(size_t i=0; i < genes_nb; ++i) {
        full.select(i);
    }
    assert(not neighborhood.hasNeighbor(full));

}
