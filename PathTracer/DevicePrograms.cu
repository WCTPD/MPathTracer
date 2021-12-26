#include <optix_device.h>
#include <iostream>
#include <optix.h>
#include <curand_kernel.h>
#include "LaunchParams.h"

namespace pt {

    extern "C" __constant__ LaunchParams optixLaunchParams;

    enum { RAY_TYPE_RADIANCE, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT };

    static __forceinline__ __device__ void init_state(radiancePRD* prd, unsigned long long seed)
    {
        curandState_t state;
        prd->state = &state;
        curand_init(seed, threadIdx.x, 0, prd->state);
    }

    static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int occluded = 0u;
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded);
        return occluded;
    }

    static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
    {
        optixSetPayload_0(static_cast<unsigned int>(occluded));
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();


        const int   primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const vec3f& A = sbtData.vertex[index.x];
        const vec3f& B = sbtData.vertex[index.y];
        const vec3f& C = sbtData.vertex[index.z];
        vec3f Ng = normalize(cross(B - A, C - A));
        const vec3f ray_dir = optixGetWorldRayDirection();


        vec3f ray_origin = optixGetWorldRayOrigin();
        vec3f P = ray_origin + optixGetRayTmax() * ray_dir;
        vec3f N = dot(Ng, -ray_dir) < 0 ? -Ng : Ng; // faceforward(Ng, -ray_dir, Ng);

        
        radiancePRD* prd = getPRD<radiancePRD>();
        curandState_t* state = prd->state;

        if (prd->countEmitted) {
            prd->emitted = sbtData.emission;
        }
        else {
            prd->emitted = vec3f(0.f);
        }
		prd->countEmitted = false;
        // vec3f wo = optixDirectCall<vec3f, const radiancePRD*, const vec3f&, const TriangleMeshSBTData&>(
        //     sbtData.sample_id,
        //     prd,
        //     N,
        //     sbtData
        // );
        vec3f wo = sampleHemiSphere(-ray_dir, N, state);
        prd->origin = P;
        prd->direction = normalize(wo);
        // prd->attenuation *= sbtData.albedo;

        // sample light
		float e1        = getRandomFloat(state), e2 = getRandomFloat(state);
		const Light light     = optixLaunchParams.light;
		const vec3f light_pos = light.corner + light.v1 * e1 + light.v2 * e2;
		const vec3f Li        = light.emission;
		const vec3f L_dir     = normalize(light_pos - P);
		float L_distance      = length(light_pos - P);
		float LDotN           = dot(L_dir, N);
		float LnDotL          = -dot(light.normal, L_dir);
        vec3f L_light = vec3f(0.f);

        auto pdf = optixDirectCall<float, const radiancePRD*, const vec3f&, const vec3f&, const TriangleMeshSBTData&>(
                sbtData.pdf_id,
                prd,
                wo,
                N,
                sbtData
            );
        auto eval = optixDirectCall<vec3f, const radiancePRD*, const vec3f&, const vec3f&, const TriangleMeshSBTData&>(
                sbtData.eval_id,
                prd,
                wo,
                N,
                sbtData
            );

        prd->attenuation *= sbtData.albedo;
		if (LDotN > 0 && LnDotL > 0) {
			const bool occluded = traceOcclusion(
				optixLaunchParams.traversable,
				P,
				L_dir,
				0.01f,
				L_distance - 0.01f
			);

			if (!occluded) {
				float A = length(cross(light.v1, light.v2));
				// float pdf_light = 1.f / A;
				//vec3f fr = 1.f / M_PI;
				// vec3f L_light = (Li * A * LDotN * LnDotL) / (L_distance * L_distance * M_PI);
                
				L_light = (A * LDotN * LnDotL) / (L_distance * L_distance * M_PI);
			}
		}
        prd->radiance += Li * L_light * eval / pdf;

    }

    extern "C" __global__ void __closesthit__occlusion()
    {
        setPayloadOcclusion(true);
    }

    extern "C" __global__ void __anyhit__radiance()
    {
        // not used
    }

    extern "C" __global__ void __miss__radiance()
    {
        MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
        radiancePRD* prd = getPRD<radiancePRD>();
        prd->radiance = data->bg_color;
        prd->done = true;
    }

    extern "C" __global__ void __miss__occlusion()
    {

    }

    extern "C" __global__ void __raygen__renderFrame()
    {
        const int w = optixLaunchParams.frame.size.x;
        const int h = optixLaunchParams.frame.size.y;
        const auto& camera = optixLaunchParams.camera;
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const int spp = optixLaunchParams.spp;
        const unsigned int subframe_index = optixLaunchParams.subframe_index;
        uint32_t u0, u1;
        radiancePRD prd;
		packPointer(&prd, u0, u1);
        unsigned long long seed = blockIdx.x * blockDim.x + threadIdx.x;
        init_state(&prd, seed);

        vec3f result(0.f);
        for (int i = 0; i < spp; i++) {
			const vec2f subpixel_jitter(getRandomFloat(prd.state) - 0.5f, getRandomFloat(prd.state) - 0.5f);
			const vec2f screen(vec2f(ix + 0.5f + subpixel_jitter.x, iy + subpixel_jitter.y + 0.5f) 
                                / vec2f(optixLaunchParams.frame.size));
			vec3f rayDir = normalize(camera.dir + (screen.y - 0.5f) * camera.vertical
                                    + (screen.x - 0.5f) * camera.horizontal);
			vec3f origin = camera.pos;
			prd.emitted = vec3f(0.f);
			prd.radiance = vec3f(0.f);
			prd.attenuation = vec3f(1.f);
			prd.countEmitted = true;
			prd.done = false;
            int Depth = 0;
            for (;;) {
                optixTrace(
                    optixLaunchParams.traversable,
                    origin,
                    rayDir,
                    0.01f,
                    1e16f,
                    0.0f,
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_RADIANCE,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_RADIANCE,
                    u0, u1
                );

                result += prd.emitted;
                result += prd.radiance * (prd.attenuation);
                /*if (prd.done || getRandomFloat(prd.state) > optixLaunchParams.P_RR)
                    break;*/
                if (prd.done || Depth > 6)
                    break;
                rayDir = prd.direction;
                origin = prd.origin;
                Depth++;
            }
        }

        vec3f pixel_color = result / static_cast<float>(spp);
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

       /* if (subframe_index > 0) {
            const float a = 1.f / static_cast<float>(subframe_index + 1);
            const vec3f accum_color_prev = optixLaunchParams.frame.accum_color[fbIndex];
            accum_color = lerp(accum_color_prev, accum_color, a);
        }*/

        vec4f rgba(pixel_color, 1.f);

        // and write to frame buffer ...
        //optixLaunchParams.frame.accum_color[fbIndex] = accum_color;
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;

    }

    extern "C" __device__ vec3f __direct_callable__lambertian_sample(
        const radiancePRD* prd,
        const vec3f& surface_noraml,
        const TriangleMeshSBTData& sbt
    )
    {
        auto scattered = sampleHemiSphere(prd->direction, surface_noraml, prd->state);
        return toWorld(surface_noraml, scattered);
    }

    extern "C" __device__ float __direct_callable__lambertian_pdf(
        const radiancePRD* prd,
        const vec3f& scattered,
        const vec3f& surface_noraml,
        const TriangleMeshSBTData& sbt
    )
    {
        return 0.5f / M_PI;
    }

    extern "C" __device__ vec3f __direct_callable__lambertian_eval(
        const radiancePRD* prd,
        const vec3f& scattered,
        const vec3f& surface_noraml,
        const TriangleMeshSBTData& sbt
    )
    {
        return sbt.albedo / M_PI;
    }

}
