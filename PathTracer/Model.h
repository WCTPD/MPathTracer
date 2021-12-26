#pragma once

#include "gdt/math/AffineSpace.h"
#include "Material.h"
#include <vector>
#include <memory>
namespace pt {

    using namespace gdt;
    using std::shared_ptr;

    struct Triangle {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        // material data:
        // vec3f diffuse;
        // vec3f emission;
        shared_ptr<Material> material;
    };

    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
        }

        std::vector<Triangle*> meshes;

        void Add(Model* model);
    };

    Model* loadOBJ(const std::string& objFile, shared_ptr<Material> mat);

}