default: all


###############################
#        CUDA settings        #
###############################

# Location of the CUDA Toolkit
# CUDA_PATH ?= /opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/11.8/
# CUDA_PATH ?= /lus/swift/soft/compilers/cudatoolkit/cuda-11.4.4/
# [2023/05/14]
# NOTE: Need to use 12.0 if want to use gcc versions later than 11.

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)

# host compiler
HOST_COMPILER ?= g++-12
NVCC          := ~/clang++
HOST_COMPILER  = ~/clang++

# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc
# [2023/04/10] NOTE: -I$(CUDA_PATH)/samples/common/inc is for helper_cuda.h
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_PATH)/lib64
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

# [2022/12/10]
# NOTE: $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) does not need to be appended at the 
# 		linking stage if the linker is invoked by NVCC; they do needed to be 
# 		appended if the linker is invoked by HOST_COMPILER.

###############################
#        CUDA settings        #
###############################


CPP_DIR_SRC := ../../src/cpp
CPP_DIR_OBJ := ../obj/cpp
CPP_DIR_DEP := ../obj/cpp

CPP_SRCS_LOCAL := $(wildcard $(CPP_DIR_SRC)/*.cpp)
CPP_OBJS_LOCAL := $(patsubst $(CPP_DIR_SRC)/%.cpp,$(CPP_DIR_OBJ)/%.o,$(CPP_SRCS_LOCAL))
CPP_DEPS_LOCAL := $(patsubst $(CPP_DIR_SRC)/%.cpp,$(CPP_DIR_DEP)/%.d,$(CPP_SRCS_LOCAL))


CPP_DIR_SRC_REMOTE := ../test_src_remote/cpp
CPP_DIR_OBJ_REMOTE := ../test_obj_remote/cpp
CPP_DIR_DEP_REMOTE := ../test_obj_remote/cpp

CPP_SRCS_REMOTE := $(wildcard $(CPP_DIR_SRC_REMOTE)/*.cpp)
CPP_OBJS_REMOTE := $(patsubst $(CPP_DIR_SRC_REMOTE)/%.cpp,$(CPP_DIR_OBJ_REMOTE)/%.o,$(CPP_SRCS_REMOTE))
CPP_DEPS_REMOTE := $(patsubst $(CPP_DIR_SRC_REMOTE)/%.cpp,$(CPP_DIR_DEP_REMOTE)/%.d,$(CPP_SRCS_REMOTE))


CPP_DIR_SRC_MAIN := ../../src/cpp_main
CPP_DIR_OBJ_MAIN := ../obj/cpp_main
CPP_DIR_DEP_MAIN := ../obj/cpp_main

CPP_SRCS_LOCAL_MAIN := $(wildcard $(CPP_DIR_SRC_MAIN)/*.cpp)
CPP_OBJS_LOCAL_MAIN := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_DIR_OBJ_MAIN)/%.o,$(CPP_SRCS_LOCAL_MAIN))
CPP_DEPS_LOCAL_MAIN := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_DIR_DEP_MAIN)/%.d,$(CPP_SRCS_LOCAL_MAIN))



CUDA_DIR_SRC := ../../src/cuda
CUDA_DIR_OBJ := ../obj/cuda
CUDA_DIR_DEP := ../obj/cuda

CUDA_SRCS_LOCAL := $(wildcard $(CUDA_DIR_SRC)/*.cu)
CUDA_OBJS_LOCAL := $(patsubst $(CUDA_DIR_SRC)/%.cu,$(CUDA_DIR_OBJ)/%.o,$(CUDA_SRCS_LOCAL))
CUDA_DEPS_LOCAL := $(patsubst $(CUDA_DIR_SRC)/%.cu,$(CUDA_DIR_DEP)/%.d,$(CUDA_SRCS_LOCAL))


CUDA_DIR_SRC_REMOTE := ../test_src_remote/cuda
CUDA_DIR_OBJ_REMOTE := ../test_obj_remote/cuda
CUDA_DIR_DEP_REMOTE := ../test_obj_remote/cuda

CUDA_SRCS_REMOTE := $(filter-out %main.cu,$(wildcard $(CUDA_DIR_SRC_REMOTE)/*.cu))
CUDA_OBJS_REMOTE := $(patsubst $(CUDA_DIR_SRC_REMOTE)/%.cu,$(CUDA_DIR_OBJ_REMOTE)/%.o,$(CUDA_SRCS_REMOTE))
CUDA_DEPS_REMOTE := $(patsubst $(CUDA_DIR_SRC_REMOTE)/%.cu,$(CUDA_DIR_DEP_REMOTE)/%.d,$(CUDA_SRCS_REMOTE))

# [2023/04/10]
# NOTE: We filter out main.cu from CUDA_SRCS_REMOTE but not main.cpp from CPP_SRCS_REMOTE. This is because
#       we have assumed that the main function is placed in main.cu.


CUDA_DIR_SRC_MAIN := ../../src/cuda_main
CUDA_DIR_OBJ_MAIN := ../obj/cuda_main
CUDA_DIR_DEP_MAIN := ../obj/cuda_main

CUDA_SRCS_LOCAL_MAIN := $(wildcard $(CUDA_DIR_SRC_MAIN)/*.cu)
CUDA_OBJS_LOCAL_MAIN := $(patsubst $(CUDA_DIR_SRC_MAIN)/%.cu,$(CUDA_DIR_OBJ_MAIN)/%.o,$(CUDA_SRCS_LOCAL_MAIN))
CUDA_DEPS_LOCAL_MAIN := $(patsubst $(CUDA_DIR_SRC_MAIN)/%.cu,$(CUDA_DIR_DEP_MAIN)/%.d,$(CUDA_SRCS_LOCAL_MAIN))


INC_SUBDIR = linear_memory helper storage # small_matrix
INC_FLAGS  = $(addprefix -I$(STOCKYARD)/github/common_headers/, $(INC_SUBDIR))

INC_FLAGS_HOST  = -I$(CPP_DIR_SRC) -I$(CPP_DIR_SRC_REMOTE) $(INC_FLAGS) -I$(CUDA_DIR_SRC_MAIN) 
# [2023/06/10]
# NOTE: We add CUDA_DIR_SRC_MAIN to the host compiler include path above so that 
#       each individual translation unit can find namespace_type.cuh .

INC_FLAGS_NVCC  = -I$(CPP_DIR_SRC) -I$(CPP_DIR_SRC_REMOTE)
INC_FLAGS_NVCC += -I$(CUDA_DIR_SRC) -I$(CUDA_DIR_SRC_REMOTE) -I$(CUDA_DIR_SRC_MAIN) 


# [2022/12/10] NOTE: Using c++20 below seems to slow code down.
CPPFLAGS_HOST  = -O3 -std=c++20
CPPFLAGS_HOST += $(INC_FLAGS_HOST)
# CPPFLAGS_HOST += -I../../../../ext_library/eigen \
# 				 -I../../../../ext_library/eigen/unsupported
# CPPFLAGS_HOST += -I/scratch/tacc/apps/intel19/eigen/3.4.0/include/eigen3/ \
# 			     -I/scratch/tacc/apps/intel19/eigen/3.4.0/include/eigen3/unsupported
# [2022/12/10]
# NOTE: -I../src/cuda is not needed for CPPFLAGS_HOST currently because currently 
# 		the cpp files do not try to #include the header files in the cuda folder.
#
# [2022/12/10]
# NOTE: The eigen library in /scratch is "installed" on lonestar6 as a module. 
# 		For some reason, module load eigen is only allowed for intel compiler.
# 		But since we know eigen is template only, providing the above path will
# 		make it work for gcc as well. In this way, we don't have to keep a copy
# 		of eigen in the local (user) folder.
# 		    The drawback is that the above "installation" paths may change.
# 		
# 		When using intel compiler, can do module load eigen, and use
# 			CPPFLAGS_HOST += -I${TACC_EIGEN_INC} -I${TACC_EIGEN_INC}/unsupported
# 		instead of the absolute path.
#
#		However, intel compiler throws out a license error every once a while. 
# 		We probably want to stay with gcc. Also, if there is any difference in 
# 		performance, gcc compiled program may be faster.
#
#		Unrelated but along the line, impi implementation still gives an error on
# 		MPI_finalize, presumably has to do with using shared memory in MPI. 

# NOTE: on https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html , it is warned 
# 		that using the flag -funroll-all-loops usually makes programs run more slowly; 
# 		however, since we know that the loops in this code are almost always very 
# 		large, it is likely that this flag actually helps. (And indeed it seems so.)


CPPFLAGS_NVCC = $(addprefix -Xcompiler , $(CPPFLAGS_HOST))
CPPFLAGS_NVCC += $(INC_FLAGS_NVCC)
# [2022/12/10] NOTE: It's surprising that the above include paths is 
# 					 not needed for nvcc to compile.
CPPFLAGS_NVCC += -std=c++20
# [2022/12/10] NOTE: nvcc version 11.4 still does not support c++20. 
# 					 We can mix 20 and 17, but it seems to be slower.
CPPFLAGS_NVCC += $(CUDA_INC_DIR)
# [2023/04/10]
# NOTE: -I$(CUDA_PATH)/samples/common/inc from $(CUDA_INC_DIR) seems to be needed
#       to #include <helper_cuda.h>. -I$(CUDA_PATH)/include may be omitted.
# CPPFLAGS_NVCC += -arch=sm_80
# [2023/05/14]
# NOTE: Need to specify -arch=sm_80. Otherwise, nvcc gives an error like " more 
#       than one constructor applies to convert from "int" to "__nv_bfloat16". "
# CPPFLAGS_NVCC += -rdc=true
# [2023/07/16]
# NOTE: The above flag (-rdc=true) is needed for "inline __device__" variables.
#       If not included, the compile-time error message says 
#           "An inline __device__/__constant__/__managed__ variable must 
#            have internal linkage when the program is compiled in whole 
#            program mode (-rdc=false)"
#       "rdc" stands for "relocatable device code". A discussion on rdc can be 
#       found at "https://forums.developer.nvidia.com/t/the-cost-of-relocatable
#       -device-code-rdc-true/47665"
#       
#       Though haven't tested yet, I believe -rdc=true will also be needed if 
#       using the "extern" mechanism for global variables.

# DPPFLAGS: flags for dependencies
DPPFLAGS = -MMD -MP -MF "$(@:%.o=%.d)" -MT "$(@)"
# [2022/12/10] 
# NOTE: With version 11.0, the nvcc compiler does not recognize the -MP flag; with 
# 		version 11.4, it does recognize and therefore, we only need 1 DPPFLAGS for 
# 		both host and device compilers.
# NOTE: For some reason, we need a space after "-MF" and "-MT" for it to work with 
# 		nvcc, while with gcc it is fine without space.

LINKER_FLAGS = -lstdc++fs
# [2022/12/10] 
# NOTE: the linker flag -lstdc++fs is needed for some older version of the compiler
#		if in utils.hpp we need to #include <experimental/filesystem> instead of 
#		#include <filesystem> and in utils.cpp we access exists () from namespace
#		std::filesystem instead of std::experimental::filesystem. On lonestar6, we 
#		can use #include <filesystem>, thus the linker flag is no longer needed.
#
# NOTE: We only need one linker flag. Usually, we invoke the linker by calling the
# 		compiler. The compiler will invoke the linker and set several linker options
# 		in the process. One can invoke the linker directly, but it's harder to get 
# 		it right. 
#
#  		Moreover, "the linker doesn't care how the objects are compiled". Compiling 
# 		using gcc or nvcc doesn't make a difference for the link as long as it has
# 		enough information to string things together. (Specifically, we don't have 
# 		to invoke the linker through the same compiler that are used to compile the 
# 		object files.)
#
# NOTE: This makefile, as well as the older version, does not go through at linking
# 		stage with mvapich2-gdr on LOGIN node, using either intel or gcc. On COMPUTE 
# 		node, it does go through. Using mvapichs (without the gdr affix), it does go 
# 		through. 
# 		    /* The complaint is libnvidia-ml.so.1 not found. 
# 			   "find / -name "*libnvidia-ml.so*" 2>/dev/null" on the login node reveals
# 		       that there is an .so version, but not an .so.1 version; there is a .so 
# 			   version on compute node in /usr/lib64
# 			*/
# 		Moreover, it seems that using mvapich is actually faster than mvapich2-gdr.
# 		mvapich2-gdr probably won't give real benefit unless we use multiple gpus.
# 		I have since updated the default module to gcc + mvapich2 combination.
# LINKER_FLAGS += -arch=sm_80
# [2023/07/16]
# NOTE: Not sure if it's because of adding -rdc=true to the compiler flag, without the
#       above linker flag added, there is a warning: 
#       		nvlink warning : SM Arch ('sm_52') not found in ...

# Additional compiler flag
CPPFLAGS_HOST += # -g  # -g is needed for using gdb
CPPFLAGS_NVCC += # -g  # -g is needed for using gdb


MACROFLAGS =


CPP_EXE_DIR := .
# CPP_EXE_LIST := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_EXE_DIR)/%.exe,$(CPP_SRCS_LOCAL_MAIN))
# [2023/06/10]
# NOTE: Comment out the above line so that if we do make/make all/mj , the main file in cpp_main 
#       won't be compiled. /* Otherwise, since namespace_type.cuh is included, the host compiler
#       won't recognize the type __half, for example. */

CUDA_EXE_DIR := .
CUDA_EXE_LIST := $(patsubst $(CUDA_DIR_SRC_MAIN)/%.cu,$(CUDA_EXE_DIR)/%.exe,$(CUDA_SRCS_LOCAL_MAIN))


all : $(CPP_EXE_LIST) $(CUDA_EXE_LIST)
# 	@echo 'executables: $(CUDA_EXE_LIST)'
# 	@echo 'object files: $(CUDA_OBJS_LOCAL_MAIN)'


CPP_EXE_NAME := $(patsubst $(CPP_DIR_SRC_MAIN)/%.cpp,$(CPP_EXE_DIR)/%,$(CPP_SRCS_LOCAL_MAIN))
$(CPP_EXE_NAME) : $(CPP_EXE_DIR)/% : $(CPP_EXE_DIR)/%.exe

CUDA_EXE_NAME := $(patsubst $(CUDA_DIR_SRC_MAIN)/%.cu,$(CUDA_EXE_DIR)/%,$(CUDA_SRCS_LOCAL_MAIN))
$(CUDA_EXE_NAME) : $(CUDA_EXE_DIR)/% : $(CUDA_EXE_DIR)/%.exe
# [2023/01/31]
# NOTE: The above rule is defined so that we can type, e.g., "make hello" 
#       instead of "make hello.exe". The definition of CUDA_EXE_NAME is almost
#       the same to "CUDA_EXE_LIST", except without the suffix ".exe".

# [2023/04/10]
# NOTE: I think the $(CUDA_EXE_NAME) : $(CUDA_EXE_DIR)/% : $(CUDA_EXE_DIR)/%.exe syntax 
#       means that the dependence rule defined by the second ":" applies 
#       to objects in the list before the first ":" only.


# [2022/12/10]
# NOTE: When linking against (static) libraries, be sure to place the (user compiled) 
# 		object files first. Otherwise, strange linking error may occur.

# Linking
$(CPP_EXE_LIST) : $(CPP_EXE_DIR)/%.exe : $(CPP_DIR_OBJ_MAIN)/%.o $(CPP_OBJS_LOCAL) $(CPP_OBJS_REMOTE)
	@echo 'Invoking linker for target: $@'
	@echo 'main object to be linked: $<'
	@echo ' '
	$(HOST_COMPILER) -o $@ $^ $(LINKER_FLAGS)
	@echo 'Finished building target: $@'
	@echo ' '

$(CUDA_EXE_LIST) : $(CUDA_EXE_DIR)/%.exe : $(CUDA_DIR_OBJ_MAIN)/%.o $(CPP_OBJS_LOCAL) $(CPP_OBJS_REMOTE) $(CUDA_OBJS_LOCAL) $(CUDA_OBJS_REMOTE)
	@echo 'Invoking linker for target: $@'
	@echo 'main object to be linked: $<'
	@echo ' '
	$(NVCC) -o $@ $^ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(LINKER_FLAGS)
	@echo 'Finished building target: $@'
	@echo ' '

# compiling .cpp
$(CPP_OBJS_LOCAL_MAIN) : $(CPP_DIR_OBJ_MAIN)/%.o: $(CPP_DIR_SRC_MAIN)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	$(HOST_COMPILER) $(CPPFLAGS_HOST) $(DPPFLAGS) $(MACROFLAGS) -o $@ -c $<
	@echo 'Finished building: $<'
	@echo ' '

$(CPP_DIR_OBJ)/%.o: $(CPP_DIR_SRC)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	${HOST_COMPILER} $(CPPFLAGS_HOST) $(DPPFLAGS) $(MACROFLAGS) -o "$@" -c "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(CPP_DIR_OBJ_REMOTE)/%.o: $(CPP_DIR_SRC_REMOTE)/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	${HOST_COMPILER} $(CPPFLAGS_HOST) $(DPPFLAGS) $(MACROFLAGS) -o "$@" -c "$<"
	@echo 'Finished building: $<'
	@echo ' '

# Compiling .cu
$(CUDA_OBJS_LOCAL_MAIN) : $(CUDA_DIR_OBJ_MAIN)/%.o: $(CUDA_DIR_SRC_MAIN)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking gcc Compiler'
	$(NVCC) $(CPPFLAGS_NVCC) $(DPPFLAGS) $(MACROFLAGS) -o $@ -c $<
	@echo 'Finished building: $<'
	@echo ' '

$(CUDA_DIR_OBJ)/%.o: $(CUDA_DIR_SRC)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking nvcc Compiler'
	$(NVCC) $(CPPFLAGS_NVCC) $(DPPFLAGS) $(MACROFLAGS) -o $@ -c $<
	@echo 'Finished building: $<'
	@echo ' '

$(CUDA_DIR_OBJ_REMOTE)/%.o: $(CUDA_DIR_SRC_REMOTE)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking nvcc Compiler'
	$(NVCC) $(CPPFLAGS_NVCC) $(DPPFLAGS) $(MACROFLAGS) -o $@ -c $<
	@echo 'Finished building: $<'
	@echo ' '


-include $(CPP_OBJS_LOCAL:.o=.d) $(CUDA_OBJS_LOCAL:.o=.d) $(CPP_OBJS_LOCAL_MAIN:.o=.d) $(CPP_OBJS_REMOTE:.o=.d) $(CUDA_OBJS_REMOTE:.o=.d) $(CUDA_OBJS_LOCAL_MAIN:.o=.d)

# Other Targets
RM := rm -rf

clm : clean_main
# [2023/01/31] NOTE: clm is shorthand for clean_main

clean_main::
	-$(RM) $(CPP_DEPS_LOCAL_MAIN) $(CPP_OBJS_LOCAL_MAIN) $(CPP_EXE_LIST)
	-$(RM) $(CUDA_DEPS_LOCAL_MAIN) $(CUDA_OBJS_LOCAL_MAIN) $(CUDA_EXE_LIST)
	-@echo ' '

clean:
	-$(RM) $(CPP_DEPS_LOCAL) $(CPP_OBJS_LOCAL) $(CUDA_DEPS_LOCAL) $(CUDA_OBJS_LOCAL) \
		   $(CPP_DEPS_REMOTE) $(CPP_OBJS_REMOTE) $(CUDA_DEPS_REMOTE) $(CUDA_OBJS_REMOTE) \
		   $(CPP_DEPS_LOCAL_MAIN) $(CPP_OBJS_LOCAL_MAIN) $(CUDA_DEPS_LOCAL_MAIN) $(CUDA_OBJS_LOCAL_MAIN) \
		   $(CPP_EXE_LIST) $(CUDA_EXE_LIST)
	-@echo ' '

.PHONY: all clean dependents
