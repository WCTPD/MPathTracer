cmake_minimum_required (VERSION 3.8)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project ("PathTracer")


set(CMAKE_CXX_STANDARD 17)

set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

# set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
# include(${gdt_dir}/cmake/configure_glfw.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE LEGACY)
if (WIN32)
    #  set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
    set(glfw_dir ${PROJECT_SOURCE_DIR}/common/3rdParty/glfw/)
    include_directories(${glfw_dir}/include)
    add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
    find_package(glfw3 REQUIRED)
endif()
include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)
# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

add_subdirectory ("PathTracer")
