#pragma once
#include "gdt/math/AffineSpace.h" 

namespace pt {

    using namespace gdt;

    enum class m_type {
        LAMBERTIAN,
        METAL,
        LIGHT,
        MICROFACET
    };

    class Material {
    public:
        virtual ~Material() {} 
        m_type type;
    };

    class Lambertian : public Material {
    public:
        Lambertian(const vec3f _albedo) : albedo(_albedo) { type = m_type::LAMBERTIAN; }
        vec3f albedo;
    };

    class Diffuse_light : public Material {
    public:
        Diffuse_light(const vec3f emit)
            : emit(emit)
        {
            type = m_type::LIGHT;
        }
        vec3f emit;
    };

    class Metal : public Material {
    public:
        Metal(const vec3f _albedo, double _roughness)
            : albedo(_albedo), roughness(_roughness)
        {
            type = m_type::METAL;
        }
        vec3f albedo;
        double roughness;
    };

}