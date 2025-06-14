# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build

# Include any dependencies generated for this target.
include CMakeFiles/ADCensusStereo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ADCensusStereo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ADCensusStereo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ADCensusStereo.dir/flags.make

CMakeFiles/ADCensusStereo.dir/ad_util.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/ad_util.cu.o: ../ad_util.cu
CMakeFiles/ADCensusStereo.dir/ad_util.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/ADCensusStereo.dir/ad_util.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/ad_util.cu.o -MF CMakeFiles/ADCensusStereo.dir/ad_util.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/ad_util.cu -o CMakeFiles/ADCensusStereo.dir/ad_util.cu.o

CMakeFiles/ADCensusStereo.dir/ad_util.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/ad_util.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/ad_util.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/ad_util.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o: ../adcensus_stereo.cu
CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o -MF CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/adcensus_stereo.cu -o CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o

CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o: ../aggregation_util.cu
CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o -MF CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/aggregation_util.cu -o CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o

CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ADCensusStereo.dir/common_util.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/common_util.cu.o: ../common_util.cu
CMakeFiles/ADCensusStereo.dir/common_util.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/ADCensusStereo.dir/common_util.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/common_util.cu.o -MF CMakeFiles/ADCensusStereo.dir/common_util.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/common_util.cu -o CMakeFiles/ADCensusStereo.dir/common_util.cu.o

CMakeFiles/ADCensusStereo.dir/common_util.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/common_util.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/common_util.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/common_util.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ADCensusStereo.dir/main.cpp.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/main.cpp.o: ../main.cpp
CMakeFiles/ADCensusStereo.dir/main.cpp.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ADCensusStereo.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/main.cpp.o -MF CMakeFiles/ADCensusStereo.dir/main.cpp.o.d -o CMakeFiles/ADCensusStereo.dir/main.cpp.o -c /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/main.cpp

CMakeFiles/ADCensusStereo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ADCensusStereo.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/main.cpp > CMakeFiles/ADCensusStereo.dir/main.cpp.i

CMakeFiles/ADCensusStereo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ADCensusStereo.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/main.cpp -o CMakeFiles/ADCensusStereo.dir/main.cpp.s

CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o: ../multistep_refine.cu
CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o -MF CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/multistep_refine.cu -o CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o

CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o: CMakeFiles/ADCensusStereo.dir/flags.make
CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o: ../scanline_optimize.cu
CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o: CMakeFiles/ADCensusStereo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o"
	/usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o -MF CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o.d -x cu -dc /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/scanline_optimize.cu -o CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o

CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target ADCensusStereo
ADCensusStereo_OBJECTS = \
"CMakeFiles/ADCensusStereo.dir/ad_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o" \
"CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/common_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/main.cpp.o" \
"CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o" \
"CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o"

# External object files for target ADCensusStereo
ADCensusStereo_EXTERNAL_OBJECTS =

CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/ad_util.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/common_util.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/main.cpp.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/build.make
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: /usr/local/cuda-11.8/lib64/libcudart.so
CMakeFiles/ADCensusStereo.dir/cmake_device_link.o: CMakeFiles/ADCensusStereo.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CUDA device code CMakeFiles/ADCensusStereo.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ADCensusStereo.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ADCensusStereo.dir/build: CMakeFiles/ADCensusStereo.dir/cmake_device_link.o
.PHONY : CMakeFiles/ADCensusStereo.dir/build

# Object files for target ADCensusStereo
ADCensusStereo_OBJECTS = \
"CMakeFiles/ADCensusStereo.dir/ad_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o" \
"CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/common_util.cu.o" \
"CMakeFiles/ADCensusStereo.dir/main.cpp.o" \
"CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o" \
"CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o"

# External object files for target ADCensusStereo
ADCensusStereo_EXTERNAL_OBJECTS =

bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/ad_util.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/adcensus_stereo.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/aggregation_util.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/common_util.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/main.cpp.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/multistep_refine.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/scanline_optimize.cu.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/build.make
bin/ADCensusStereo: /usr/local/cuda-11.8/lib64/libcudart.so
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/cmake_device_link.o
bin/ADCensusStereo: CMakeFiles/ADCensusStereo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable bin/ADCensusStereo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ADCensusStereo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ADCensusStereo.dir/build: bin/ADCensusStereo
.PHONY : CMakeFiles/ADCensusStereo.dir/build

CMakeFiles/ADCensusStereo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ADCensusStereo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ADCensusStereo.dir/clean

CMakeFiles/ADCensusStereo.dir/depend:
	cd /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build /media/zhangwei/新加卷/MyCode/ADCensus_CUDA-main/build/CMakeFiles/ADCensusStereo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ADCensusStereo.dir/depend

