# include <mpi.h>
# include <stdio.h>
# include <string.h>

int main(int argc, char **argv)
{
    int rank, size;
    char someString[] = "Can haz cheezburgerz?";

    MPI_Init(&argc, &argv);

    MPI_Comm_rank( MPI_COMM_WORLD, & rank );
    MPI_Comm_size( MPI_COMM_WORLD, & size );

    if ( rank == 0 )
    {
        int n = 42;
        int i;
        for( i = 1; i < size; ++i)
        {
            MPI_Send( &n, 1, MPI_INT, i, 0, MPI_COMM_WORLD );
            MPI_Send( &someString, strlen( someString )+1, MPI_CHAR, i, 0, MPI_COMM_WORLD );
        }
    } else {
        char buffer[ 128 ];
        int received;
        MPI_Status stat;
        MPI_Recv( &received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat );
        printf( "[Worker] Number : %d\n", received );
        MPI_Recv( buffer, 128, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat );
        printf( "[Worker] String : %s\n", buffer );
    }

    MPI_Finalize();
}
