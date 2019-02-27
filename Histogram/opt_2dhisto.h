#ifndef OPT_KERNEL
#define OPT_KERNEL


//__global__ void histogram256Kernel (uint32_t *d_Result, uint32_t *d_Data, int dataN); 

void opt_2dhisto (uint32_t *d_Result, uint32_t *d_Data, int size_input); 

// set up
uint32_t* allocateHist(uint32_t *new_bins, int size_hist); 
uint32_t* allocateInput(uint32_t *new_input, int size_input); 

//tear down
void free_GPU_vars (uint32_t *d_Data ,uint32_t *d_Result);
void copyHistToCPU (uint32_t* new_bins, uint32_t* d_Result, int size_hist);
void free_GPU_vars (uint32_t *d_Data ,uint32_t *d_Result);



#endif
