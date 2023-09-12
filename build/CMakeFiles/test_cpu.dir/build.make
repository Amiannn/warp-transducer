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
include CMakeFiles/test_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_cpu.dir/flags.make

CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o: CMakeFiles/test_cpu.dir/flags.make
CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o: /share/nas165/amian/experiments/speech/warp-transducer/tests/test_cpu.cpp
CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o: CMakeFiles/test_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o -MF CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o.d -o CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o -c /share/nas165/amian/experiments/speech/warp-transducer/tests/test_cpu.cpp

CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /share/nas165/amian/experiments/speech/warp-transducer/tests/test_cpu.cpp > CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.i

CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /share/nas165/amian/experiments/speech/warp-transducer/tests/test_cpu.cpp -o CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.s

CMakeFiles/test_cpu.dir/tests/random.cpp.o: CMakeFiles/test_cpu.dir/flags.make
CMakeFiles/test_cpu.dir/tests/random.cpp.o: /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp
CMakeFiles/test_cpu.dir/tests/random.cpp.o: CMakeFiles/test_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_cpu.dir/tests/random.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_cpu.dir/tests/random.cpp.o -MF CMakeFiles/test_cpu.dir/tests/random.cpp.o.d -o CMakeFiles/test_cpu.dir/tests/random.cpp.o -c /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp

CMakeFiles/test_cpu.dir/tests/random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_cpu.dir/tests/random.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp > CMakeFiles/test_cpu.dir/tests/random.cpp.i

CMakeFiles/test_cpu.dir/tests/random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_cpu.dir/tests/random.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /share/nas165/amian/experiments/speech/warp-transducer/tests/random.cpp -o CMakeFiles/test_cpu.dir/tests/random.cpp.s

# Object files for target test_cpu
test_cpu_OBJECTS = \
"CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o" \
"CMakeFiles/test_cpu.dir/tests/random.cpp.o"

# External object files for target test_cpu
test_cpu_EXTERNAL_OBJECTS =

test_cpu: CMakeFiles/test_cpu.dir/tests/test_cpu.cpp.o
test_cpu: CMakeFiles/test_cpu.dir/tests/random.cpp.o
test_cpu: CMakeFiles/test_cpu.dir/build.make
test_cpu: libwarprnnt.so
test_cpu: /usr/local/cuda/lib64/libcudart_static.a
test_cpu: /usr/lib/x86_64-linux-gnu/librt.so
test_cpu: CMakeFiles/test_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_cpu.dir/build: test_cpu
.PHONY : CMakeFiles/test_cpu.dir/build

CMakeFiles/test_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_cpu.dir/clean

CMakeFiles/test_cpu.dir/depend:
	cd /share/nas165/amian/experiments/speech/warp-transducer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /share/nas165/amian/experiments/speech/warp-transducer /share/nas165/amian/experiments/speech/warp-transducer /share/nas165/amian/experiments/speech/warp-transducer/build /share/nas165/amian/experiments/speech/warp-transducer/build /share/nas165/amian/experiments/speech/warp-transducer/build/CMakeFiles/test_cpu.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_cpu.dir/depend

