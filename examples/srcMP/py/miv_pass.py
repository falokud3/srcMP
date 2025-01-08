''' Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
     Based on cetus TooSmall.c'''


def main():

    a = [0] * 1000
    b = [0] * 1000

    for i in range(1000):
        a[i] = a[i] + 2

    return 0