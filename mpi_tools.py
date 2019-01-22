from math import ceil

from mpi4py import MPI

def get_mpi_rank():
    comm = MPI.COMM_WORLD
    return comm.Get_rank()

def get_mpi_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()

def mpi_barrier():
    comm = MPI.COMM_WORLD
    return comm.Barrier()

def mpi_broadcast(data):
    comm = MPI.COMM_WORLD
    return comm.bcast(data, root=0)	

def sublist_for_a_process(full_list):
    rank = get_mpi_rank()
    size = get_mpi_size()

    n = len(full_list)
    n_per_process = ceil(n / size)

    return full_list[rank*n_per_process:(rank+1)*n_per_process]
