ROCM_PATH?= $(wildcard /opt/rocm)

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
    HIP_PATH=../../..
endif

ROCSPARSE_INCLUDE = ${ROCM_PATH}/rocsparse/include
ROCSPARSE_LIB = ${ROCM_PATH}/rocsparse/lib/

HIPCC=$(HIP_PATH)/bin/hipcc

gpu = 0
ifeq ($(gpu),0)
	EXECUTABLE = Csrsparse
	CXXFLAGS = -O2 -I ./ 
	LDFLAGS = 
else ifeq ($(gpu),1)
	EXECUTABLE = Csrsparse_rocsparse
	CXXFLAGS = -Dgpu -O2 -I ./ -I ${ROCSPARSE_INCLUDE}
	LDFLAGS = -L ${ROCSPARSE_LIB} -lrocsparse
endif

# targets
.PHONY: clean all
all: $(EXECUTABLE) 

## spgemm building start.
### configs
GENERATED_HEADER_PATH = ./
SPGEMM_LIB_INC_PATH = -I./ -I$(GENERATED_HEADER_PATH) -I./sort -I./sort/api -I./sort/hip-block-radix -I./spgemm -I./spgemm/api/ -I./spgemm/common -I./spgemm/hip-dense -I./spgemm/hip-kokkos
SPGEMM_LIB_HIPCC_FLAGS = $(SPGEMM_LIB_INC_PATH) -O2 -std=gnu++14 # --save-temps
SORT_LIB_HIPCC_FLAGS = $(SPGEMM_LIB_INC_PATH) -O2 -std=gnu++14

SPGEMM_BIN_NAME = spgemm-hip
SPGEMM_KERNEL_LIB_NAME = spgemm-kernels
SORT_KERNEL_LIB_NAME = sort-kernels
SPGEMM_KERNEL_LIB_PATH = libspgemm-kernels.a
SORT_KERNEL_LIB_PATH = libsort-kernels.a
### configs end

### sort lib
$(SORT_KERNEL_LIB_PATH): sort_api_sort_hip_api.cc.o
	ar qc $(SORT_KERNEL_LIB_PATH) sort_api_sort_hip_api.cc.o
	ranlib $(SORT_KERNEL_LIB_PATH)
sort_api_sort_hip_api.cc.o: sort/api/sort_hip_api.cc
	$(HIPCC) $(SORT_LIB_HIPCC_FLAGS) -o sort_api_sort_hip_api.cc.o -c sort/api/sort_hip_api.cc
### sort lib end
### spgemm lib
$(SPGEMM_KERNEL_LIB_PATH): spgemm_api_spgemm_hip_api.cc.o
	ar qc $(SPGEMM_KERNEL_LIB_PATH)  spgemm_api_spgemm_hip_api.cc.o
	ranlib $(SPGEMM_KERNEL_LIB_PATH)
spgemm_api_spgemm_hip_api.cc.o: spgemm/api/spgemm_hip_api.cc
	$(HIPCC) $(SORT_LIB_HIPCC_FLAGS) -o spgemm_api_spgemm_hip_api.cc.o -c spgemm/api/spgemm_hip_api.cc
### spgemm lib end
## spgemm building end.


$(EXECUTABLE): main.cpp.o  $(SORT_KERNEL_LIB_PATH) $(SPGEMM_KERNEL_LIB_PATH)
	$(HIPCC) main.cpp.o $(LDFLAGS) -o $(EXECUTABLE) -L./ $(SORT_KERNEL_LIB_PATH) $(SPGEMM_KERNEL_LIB_PATH)

main.cpp.o: main.cpp
	$(HIPCC) $(CXXFLAGS) ${SPGEMM_LIB_HIPCC_FLAGS} -c main.cpp -o main.cpp.o

clean:
	rm -f *.o
	rm -f ${EXECUTABLE}
	rm -rf $(SORT_KERNEL_LIB_PATH) $(SPGEMM_KERNEL_LIB_PATH)
