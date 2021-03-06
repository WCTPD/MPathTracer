#ifndef RENDERER_H
#define RENDERER_H

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"
#

namespace pt {
    
    class Camera {
    public:
        vec3f pos;
        vec3f at;
        vec3f up;
    };

    class Renderer
    {
    public:
        /*! constructor - performs all setup, including initializing
         optix, creates module, pipeline, programs, SBT, etc. */
        Renderer(const Model* model);

         /*! render one frame */
         void render();

         /*! resize frame buffer to given resolution */
         void resize(const vec2i& newSize);

         /*! download the rendered color buffer */
         void downloadPixels(vec4f h_pixels[]);

         /*! set camera to render with */
        void setCamera(const Camera& camera);

        void setLight();

        void update_subframe_index();
    private:
        /*! @{ CUDA device context and stream that optix pipeline will run
            on, as well as device properties for this device */
        CUcontext          cudaContext;
        CUstream           stream;
        cudaDeviceProp     deviceProps;

        //! the optix context that our pipeline will run in.
        OptixDeviceContext optixContext;

        /*! @{ the pipeline we're building */
        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions = {};

        /*! @{ the module that contains out device programs */
        OptixModule                 module;
        OptixModuleCompileOptions   moduleCompileOptions = {};

        /*! vector of all our program(group)s, and the SBT built around
           them */
        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        std::vector<OptixProgramGroup> callablePGs;
        CUDABuffer callableRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        /*! @{ our launch parameters, on the host, and the buffer to store
           them on the device */
        LaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;

        CUDABuffer colorBuffer;
        CUDABuffer accum_color_buffer;

        /*! the camera we are to render with. */
        Camera lastSetCamera;

        /*! the model we are going to trace rays against */
        const Model* model;

        /*! one buffer per input mesh */
        std::vector<CUDABuffer> vertexBuffer;
        /*! one buffer per input mesh */
        std::vector<CUDABuffer> indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;
        CUDABuffer denoisedBuffer;

        OptixDenoiser denoiser = nullptr;
        CUDABuffer    denoiserScratch;
        CUDABuffer    denoiserState;

        // ------------------------------------------------------------------
       // internal helper functions
       // ------------------------------------------------------------------

       /*! helper function that initializes optix and checks for errors */
        void initOptix();

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void createContext();

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void createModule();

        /*! does all setup for the raygen program(s) we are going to use */
        void createRaygenPrograms();

        /*! does all setup for the miss program(s) we are going to use */
        void createMissPrograms();

        /*! does all setup for the hitgroup program(s) we are going to use */
        void createHitgroupPrograms();

        void createCallablegroupPrograms();

        /*! assembles the full pipeline of all programs */
        void createPipeline();

        /*! constructs the shader binding table */
        void buildSBT();

        /*! build an acceleration structure for the given triangle mesh */
        OptixTraversableHandle buildAccel();
    };

}

#endif