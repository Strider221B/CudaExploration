.PHONY: 01 01_obj 01_obj_exe_run 02 03 clean
# Better to mention the targets in the phony section. Else if there's a file in the folder with the same name, then make will directly use the file instead of using our target.
# Also note: the name of this file must be Makefile and not something like MakeFile. 

# variables
GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_75

01:
	@$(GCC) -o 01 01.c

01_obj:
	@$(GCC) -c 01.c -o 01.o

# format -> target : pre-req. IT'll automatically run the pre-req if you call make on target.
01_obj_exe_run: 01_obj
	@$(GCC) 01.o -o 01
	@./01

02:
	@$(GCC) -o 02 02.c

03: 
	@$(NVCC) $(CUDA_FLAGS) -o 03_cu 03.cu

clean: 
	@rm -f 01 02 03_cu *.o

# How to run:
# somesh@LAPTOP-OV0CBJU0:~/Git/CudaExploration/02_cpp/05_make_files$ make 03
# somesh@LAPTOP-OV0CBJU0:~/Git/CudaExploration/02_cpp/05_make_files$ make clean
