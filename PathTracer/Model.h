#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
namespace pt {

    using namespace gdt;

    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        // material data:
        vec3f diffuse;
        vec3f emission;
    };

    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
        }

        std::vector<TriangleMesh*> meshes;

        void Add(Model* model);
    };

    Model* loadOBJ(const std::string& objFile);

}