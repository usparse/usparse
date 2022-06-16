#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <vector>
#include <hip/hip_runtime.h>
#include "./common_function.hpp"
#include "./Csrsparse.hpp"

using namespace std;


int main(int argc,char** argv)
{  
    cout << "=================================================================="<<endl;
    cout << "Instructions:"<<endl;
    cout << "[1] If using local file to creat matrix A, input args shold be: "<<endl;
    cout << "    1, CSR file path, sparsity of B, random seed of generating matrix "<<endl;
    cout << "[2] If generating matrix A randomly, input args shold be: "<<endl;
    cout << "    0, rows of A, columns of A, sparsity of A, sparsity of B, random seed of generating matrix "<<endl;

    cout << "=================================================================="<<endl;

    //Set random alpha
    srand((unsigned)time(NULL));
    alpha = (dtype)(rand()%10000);

    //Get args to Init matrix
    int FromFile = 0;
    FromFile = atoi(argv[1]);

    if(FromFile==0)
    {
        cout << "matrix A and matrix B are both generated randomly!"<<endl;
        //Size of Matrix
        A_num_rows = atol(argv[2]);
    	A_num_cols = atol(argv[3]);
    	B_num_rows = A_num_cols;
        B_num_cols = B_num_rows;

    	//sparsity
    	sparsityA = atof(argv[4]);
    	sparsityB = atof(argv[5]);
        random_seed = atol(argv[6]);
    }
    else if(FromFile==1)
    {
        cout << "matrix A is read from CSR file, while matrix B is generated randomly!"<<endl;
        char path[1024];
        strcpy(path, argv[2]);
        sparsityB = atof(argv[3]);
        random_seed = atol(argv[4]);
        
        cout << "Reading Data File ... " << endl;                        
        read_file(path);

        A_num_rows = dense_vector.size();
        A_num_cols = dense_vector.size();
        B_num_rows = A_num_cols;
        B_num_cols = B_num_rows;
    }

    //Set random seed for generating sparse matrix
    srand(random_seed);
         
    //Host Data Init
    cout << "Preparing Host Memory ... " << endl;
    create_host_data(FromFile);

    //Show Info
    cout << "marix A: " << A_num_rows << " X " << A_num_cols << "    marix B: " << B_num_rows << " X " << B_num_cols << endl;
    if(FromFile==0)
    {
        cout << "sparsity A: " << sparsityA << "    sparsity B: " << sparsityB << endl;
    }
    else
    {
        cout << "sparsity B: " << sparsityB << endl;
    }
    cout << "random seed for generating matrx : " << random_seed << endl;
    cout << "alpha = " << alpha << endl;
    cout << "=================================================================="<<endl;

    //Set Device Id for the following Calculation
    hipSetDevice(0);
 
    //Device Memory malloc + Copy 
    cout << "Preparing Device Memory ... " << endl;  
    create_deivce_data();
    cout << "=================================================================="<<endl;

    size_t C_nnz = 0;                              //None zero num of C calculated by kernel
    size_t C_nnz_compare = 0;                      //None zero num of C calculated by compare function

    csrIdxType* device_offsetC;                    //csr offset device ptr of C
    csrIdxType* device_colindexC;                  //csr column index device ptr of C
    dtype*  device_valueC;                         //csr value device ptr of C

    HIP_CHECK( hipMalloc((void**) &device_offsetC, (A_num_rows+1) * sizeof(csrIdxType)) )
    
    cout << "Runing device spgemm ... " << endl;
    int ROUND_GPU = 1;

    //Create timer_kernel
    my_timer timer_kernel;
    
    double time_use_kernel = 0.0;
    
    for(int i=0; i<ROUND_GPU; i++)
    {
        //Start timer_kernel
        timer_kernel.start();

        // Execute Device spgemm 
        call_device_spgemm(opA, opB, alpha, A_num_rows, B_num_cols, A_num_cols, 
                                A_nnz, device_offsetA, device_colindexA, device_valueA,
                                B_nnz, device_offsetB, device_colindexB, device_valueB,
                                &C_nnz, device_offsetC, &device_colindexC, &device_valueC);

        hipDeviceSynchronize();

        //Stop timer kernel 
        timer_kernel.stop();   
        
        time_use_kernel += timer_kernel.time_use;
        
        if( i==(ROUND_GPU-1) )
        {
            host_offsetC= new csrIdxType[A_num_rows+1];
            host_valueC = new dtype[C_nnz];
            HIP_CHECK( hipMemcpy(host_valueC, device_valueC, C_nnz * sizeof(dtype), hipMemcpyDeviceToHost) )
            HIP_CHECK( hipMemcpy(host_offsetC, device_offsetC, (A_num_rows+1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
        }


        HIP_CHECK( hipFree(device_colindexC) )
        HIP_CHECK( hipFree(device_valueC) )
    }
    HIP_CHECK( hipFree(device_offsetC) )

    cout << "non zero num of matrix C : " << C_nnz << endl;
    cout << "Device calculation finished!  elapsed time:" << time_use_kernel << "(us)" << endl;
    cout << "=================================================================="<<endl;

    
    #ifdef gpu
        cout << "Runing rocsparse spgemm ... " << endl;
        //Create timer_rocsparse
        my_timer timer_rocsparse;
    
        double time_use_rocsparse = 0.0;

        //Start timer_rocsparse
        timer_rocsparse.start();

        //RocSparse Calculation
        C_nnz_compare = rocsparse_spgemm(host_offsetC);

        hipDeviceSynchronize();	      

        //Stop timer_rocsparse
        timer_rocsparse.stop();

        time_use_rocsparse += timer_rocsparse.time_use;

        cout<<"Rocsparse calculation finished!  elapsed time:"<<time_use_rocsparse<<"(us)" << endl;
    #else
        cout << "Runing host spgemm ... " << endl;
        //Create timer_hostcpu
        my_timer timer_hostcpu;

        double time_use_hostcpu = 0.0;

        //Start timer_hostcpu
        timer_hostcpu.start();

        //Host Calculation
        C_nnz_compare =  host_spgemm();

        //Stop timer_hostcpu
        timer_hostcpu.stop();

        time_use_hostcpu += timer_hostcpu.time_use;

        cout<<"Host calculation finished!  elspsed time:"<<time_use_hostcpu<<"(us)"  << endl;    
    #endif
    cout << "=================================================================="<<endl;

    //Verify Result
    cout << "Runing verify ... " << endl;
    cout<<"host_valueC[C_nnz]="<<host_valueC[C_nnz-1]<<endl;  
    cout<<"host_compare_valueC[C_nnz_compare]="<<host_compare_valueC[C_nnz_compare-1]<<endl;
    cout<<"host_compare_valueC[C_nnz_compare-1]="<<host_compare_valueC[C_nnz_compare-2]<<endl; 

    // if(C_nnz != C_nnz_compare)
    // {
    //     cout << "Non zero num not the same, you kernel may not be correct!" << endl;
    //     return 0;
    // }


    verify(host_valueC, host_compare_valueC, C_nnz);


    cout << "=================================================================="<<endl;
    return 0;
}
