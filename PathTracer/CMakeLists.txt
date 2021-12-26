find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable (PathTracer 
			    ${embedded_ptx_code}
				"PathTracer.cpp" 
				"PathTracer.h"
				"LaunchParams.h"
        "tiny_obj_loader.h" 
        "Renderer.h" 
        "Renderer.cpp" 
        "CUDABuffer.h" 
        "optix7.h"  
        "Model.h" 
        "Model.cpp"
        "Material.h"
        )


target_link_libraries(PathTracer 
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  )
