#pragma once

#include "gdt/math/vec.h"
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace pt {

	using namespace gdt;

	enum {
		LAMBERTIAN_SAMPLE,
		LAMBERTIAN_PDF,
		LAMBERTIAN_EVAL,
		CALLABLE_PGS,
	}; // callable id

	struct Light {
		vec3f emission;
		vec3f corner;
		vec3f v1;
		vec3f v2;
		vec3f normal;
	};

	struct LaunchParams
	{
		struct {
			float4* colorBuffer;
			float4* accum_color;
			vec2i size;
		} frame;

		struct {
			vec3f pos;
			vec3f dir;
			vec3f horizontal;
			vec3f vertical;
		} camera;

		Light light;

		int light_samples;
		int spp;
		float P_RR;
		unsigned int subframe_index;

		OptixTraversableHandle traversable;
	};

	struct TriangleMeshSBTData {
		vec3f albedo;
		vec3f* vertex;
		vec3i* index;
		vec3f emission;
		double roughness;
		int pdf_id, sample_id, eval_id;
	};

	struct MissData {
		vec3f bg_color;
	};

	struct radiancePRD {
		vec3f       emitted;
		vec3f       radiance;
		vec3f       attenuation;
		vec3f       origin;
		vec3f       direction;
		int          countEmitted;
		int          done;
		int          pad;
		curandState_t *state;
	};

	struct Onb
	{
		__forceinline__ __device__ Onb(const vec3f& normal)
		{
			m_normal = normal;

			if (fabs(m_normal.x) > fabs(m_normal.z))
			{
				m_binormal.x = -m_normal.y;
				m_binormal.y = m_normal.x;
				m_binormal.z = 0;
			}
			else
			{
				m_binormal.x = 0;
				m_binormal.y = -m_normal.z;
				m_binormal.z = m_normal.y;
			}

			m_binormal = normalize(m_binormal);
			m_tangent = cross(m_binormal, m_normal);
		}

		__forceinline__ __device__ void inverse_transform(vec3f& p) const
		{
			p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
		}

		vec3f m_tangent;
		vec3f m_binormal;
		vec3f m_normal;
	};
    
	__forceinline__ __device__ vec3f lerp(const vec3f& a, const vec3f& b, const float t)
    {
        return a + t * (b - a);
    }

	__forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    __forceinline__ __device__ float getRandomFloat(curandState_t* state)
    {
        float a = curand_uniform(state);
        return a;
    }

    __forceinline__ __device__ float clamp(const float& lo, const float& hi, const float& v)
    {
        return max(lo, min(hi, v));
    }

    __forceinline__ __device__ vec3f toWorld(const vec3f& N, const vec3f& ray)
    {
        vec3f m_tangent;
        vec3f m_binormal;
        vec3f m_normal = N;
        if (fabs(N.x) > fabs(N.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
        return ray.x * m_tangent + ray.y * m_binormal + ray.z * m_normal;
    }

    __forceinline__ __device__ vec3f sampleHemiSphere(const vec3f& wi, const vec3f& N, curandState_t* state)
    {
        float x1 = getRandomFloat(state), x2 = getRandomFloat(state);
        float phi = 2.f * M_PI * x2;
        float r = sqrtf(x1);
        float x = r * cosf(phi);
        float y = r * sinf(phi);
        vec3f wo(x, y, sqrt(fmaxf(0.f, 1.f - x * x - y * y)));
        return toWorld(N, wo);
    }
} 