#include "psrs.h"


void random_array(int* a, int num) {
    for (int i = 0; i < num; i++) {
        srand(clock());
        a[i] = rand() % RNG_MODIFIER;
    }
}

void print_array(int* a, int num) {
    for (int i = 0; i < num; i++) {
        if (i % 20 == 0)
            printf("\n");
        printf("%d ", a[i]);
    }
    printf("\n");
}

bool check_array(int* B, int* C, int num) {
    for (int i = 0; i < num; i++) {
        if (B[i] != C[i]) {
            printf("A[%d] = %d not %d!\n", i, B[i], C[i]);
            return false;
        }
    }
    return true;
}

bool compare(const void* arg1, const void* arg2) {
    return *(int*)arg1 >= *(int*)arg2;
}

void merge(int* a, int p, int q, int r) {
    int n1 = q - p + 1;
    int n2 = r - q;
    int L[n1 + 1], R[n2 + 1];
    for (int i = 0; i < n1; i++)
        L[i] = a[p + i];
    L[i] = SIZE;
    for (int j = 0; j < n2; j++)
        R[j] = a[q + j + 1];
    R[j] = SIZE;
    i = 0, j = 0;
    for (int k = p; k <= r; k++) {
        if (L[i] <= R[j]) {
            a[k] = L[i];
            i++;
        }
        else {
            a[k] = R[j];
            j++;
        }
    }
}


void merge_sort(int* a, int p, int r) {
    if (p < r) {
        int q = (p + r) / 2;
        merge_sort(a, p, q);
        merge_sort(a, q + 1, r);
        merge(a, p, q, r);
    }
}



int main(int argc, char* argv[]) {
    int* array, * a;
    int* result, * ans;
    int num_processes, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Request request;

    if (argc != 2) {
        printf("Invalid arguments\n");
        return -1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("Size of array must be greater than 1\n");
        return -1;
    }

    int group = n / num_processes;
    int remainder = n % num_processes;
    a = (int*)malloc((group + remainder) * sizeof(int));

    if (rank == 0) {
        // sequential generation of array, performed by the master
        array = (int*)malloc((n + 2) * sizeof(int));
        result = (int*)malloc((n + 2) * sizeof(int));
        ans = (int*)malloc((n + 2) * sizeof(int));
        random_array(array, n);
        memcpy(ans, array, (n) * sizeof(int));
        qsort(ans, n, sizeof(int), compare);
    }

    int sendcounts[num_processes];
    int sdispls[num_processes];
    for (int i = 0; i < num_processes; i++) {
        sendcounts[i] = group;
        sdispls[i] = i * group;
    }
    sendcounts[num_processes - 1] = group + remainder;

    // scatter the data to the slaves
    MPI_Scatterv(array, sendcounts, sdispls, MPI_INT, a, group + remainder, MPI_INT, 0, MPI_COMM_WORLD);

    int group_len = sendcounts[rank];

    // partition the data and quick sort on each slave
    qsort(a, group_len, sizeof(int), compare);

    int samples[num_processes * num_processes];
    int s[num_processes];
    for (int i = 1; i < num_processes; i++) {
        s[i - 1] = a[i * group / num_processes];
    }

    // Use MPI_Gather to receive the data
    MPI_Gather(s, num_processes - 1, MPI_INT, samples, num_processes - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort the samples
    int pivot[num_processes];
    if (rank == 0) {
        qsort(samples, (num_processes - 1) * num_processes, sizeof(int), compare);
        for (int i = 1; i < num_processes; i++)
            pivot[i - 1] = samples[i * (num_processes - 1)];
    }
    // Broadcast the pivot
    MPI_Bcast(pivot, num_processes - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition according to the pivot
    int index = 0;
    int pcounts[num_processes];
    for (int i = 0; i < num_processes; i++)
        pcounts[i] = 0;
    pivot[num_processes - 1] = INT32_MAX;

    for (int i = 0; i < group_len; i++) {
        if (a[i] <= pivot[index])
            pcounts[index]++;
        else {
            i--;
            index++;
        }
    }

    // Alltoall call to send the rcounts
    int rcounts[num_processes];
    MPI_Alltoall(pcounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int rdispls[num_processes];
    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < num_processes; i++) {
        sdispls[i] = sdispls[i - 1] + pcounts[i - 1];
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
    }
    int totalcounts = 0;
    for (int i = 0; i < num_processes; i++)
        totalcounts += rcounts[i];

    int* b = (int*)malloc(totalcounts * sizeof(int));

    // Each process send data to to other processor
    // length and offset are included as they might be different from process to process
    MPI_Alltoallv(a, pcounts, sdispls, MPI_INT,
        b, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    merge_sort(b, 0, totalcounts - 1);

    // Gather sorted sub-array from the slaves
    MPI_Gather(&totalcounts, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rdispls[0] = 0;
    for (int i = 1; i < num_processes; i++)
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];

    MPI_Gatherv(b, totalcounts, MPI_INT, result, rcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (check_array(result, ans, n))
            printf("Finish\n");
        else
            printf("Error in sorting\n");
        free(ans);
        free(result);
        free(array);
    }
    if (b != NULL) {
        free(b);
    }
    if (a != NULL) {
        free(a);
    }

    MPI_Finalize();
    return 0;
}