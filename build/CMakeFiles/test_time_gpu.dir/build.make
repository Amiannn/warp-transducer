# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /share/homes/amian/miniconda3/envs/transducer/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /share/homes/amian/miniconda3/envs/transducer/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /share/nas165/amian/experiments/speech/warp-transducer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /share/nas165/amian/experiments/speech/warp-transducer/build

# Include any dependencies generated for this target.
include CMakeFiles/test_time_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_time_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_time_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_time_gpu.dir/flags.make

CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o: /share/nas165/amian/experiments/speech/warp-transducer/tests/test_time.cu
CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o: CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o.depend
CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o: CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o"
	cd /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests && /share/homes/amian/miniconda3/envs/transducer/lib/python3.9/site-packages/cmake/data/bin/cmake -E make_directory /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests/.
	cd /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests && /share/homes/amian/miniconda3/envs/transducer/lib/python3.9/site-packages/cmake/data/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests/./test_time_gpu_generated_test_time.cu.o -D generated_cubin_file:STRING=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests/./test_time_gpu_generated_test_time.cu.o.cubin.txt -P /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o.cmake

CMakeFiles/test_time_gpu.dir/tests/random.cpp.o: CMakeFiles/test_time_gpu.dir/flags.make
CMakeFiles/test_time_gpu.dir/tests/random.cpp.o: /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp
CMakeFiles/test_time_gpu.dir/tests/random.cpp.o: CMakeFiles/test_time_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_time_gpu.dir/tests/random.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_time_gpu.dir/tests/random.cpp.o -MF CMakeFiles/test_time_gpu.dir/tests/random.cpp.o.d -o CMakeFiles/test_time_gpu.dir/tests/random.cpp.o -c /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp

CMakeFiles/test_time_gpu.dir/tests/random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_time_gpu.dir/tests/random.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp > CMakeFiles/test_time_gpu.dir/tests/random.cpp.i

CMakeFiles/test_time_gpu.dir/tests/random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_time_gpu.dir/tests/random.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp -o CMakeFiles/test_time_gpu.dir/tests/random.cpp.s

# Object files for target test_time_gpu
test_time_gpu_OBJECTS = \
"CMakeFiles/test_time_gpu.dir/tests/random.cpp.o"

# External object files for target test_time_gpu
test_time_gpu_EXTERNAL_OBJECTS = \
"/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o"

test_time_gpu: CMakeFiles/test_time_gpu.dir/tests/random.cpp.o
test_time_gpu: CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o
test_time_gpu: CMakeFiles/test_time_gpu.dir/build.make
test_time_gpu: /usr/local/cuda/lib64/libcudart_static.a
test_time_gpu: /usr/lib/x86_64-linux-gnu/librt.so
test_time_gpu: libwarprnnt.so
test_time_gpu: /usr/local/cuda/lib64/libcurand.so
test_time_gpu: /usr/local/cuda/lib64/libcudart_static.a
test_time_gpu: /usr/lib/x86_64-linux-gnu/librt.so
test_time_gpu: CMakeFiles/test_time_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_time_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_time_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_time_gpu.dir/build: test_time_gpu
.PHONY : CMakeFiles/test_time_gpu.dir/build

CMakeFiles/test_time_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_time_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_time_gpu.dir/clean

CMakeFiles/test_time_gpu.dir/depend: CMakeFiles/test_time_gpu.dir/tests/test_time_gpu_generated_test_time.cu.o
	cd /share/nas165/amian/experiments/speech/warp-transducer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /share/nas165/amian/experiments/speech/warp-transducer /share/nas165/amian/experiments/speech/warp-transducer /share/nas165/amian/experiments/speech/warp-transducer/build /share/nas165/amian/experiments/speech/warp-transducer/build /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_time_gpu.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_time_gpu.dir/depend

