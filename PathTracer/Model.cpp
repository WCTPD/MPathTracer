#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <set>

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

namespace pt {

    int addVertex(Triangle* mesh,
        const tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
        const vec3f* normal_array = (const vec3f*)attributes.normals.data();
        const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();

        int newID = mesh->vertex.size();
        knownVertices[idx] = newID;

        mesh->vertex.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normal.size() < mesh->vertex.size())
                mesh->normal.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texcoord.size() < mesh->vertex.size())
                mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
        }

        // just for sanity's sake:
        if (mesh->texcoord.size() > 0)
            mesh->texcoord.resize(mesh->vertex.size());
        // just for sanity's sake:
        if (mesh->normal.size() > 0)
            mesh->normal.resize(mesh->vertex.size());

        return newID;
    }

    Model* loadOBJ(const std::string& objFile, shared_ptr<Material> mat)
    {
        Model* model = new Model;

        tinyobj::ObjReaderConfig reader_config;
        auto n = objFile.rfind("/");
        if (n == std::string::npos)
            reader_config.mtl_search_path = "./";
        else
            reader_config.mtl_search_path = objFile.substr(0, n);

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(objFile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyReader: " << reader.Error();
            }
            exit(-1);
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyLoader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
            const tinyobj::shape_t& shape = shapes[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                Triangle* mesh = new Triangle;

                for (int faceID = 0; faceID < (int)shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    vec3i idx(addVertex(mesh, attrib, idx0, knownVertices),
                        addVertex(mesh, attrib, idx1, knownVertices),
                        addVertex(mesh, attrib, idx2, knownVertices));
                    mesh->index.push_back(idx);
                    // mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
                    // mesh->emission = (const vec3f&)materials[materialID].emission;
                    mesh->material = mat;
                }

                if (mesh->vertex.empty())
                    delete mesh;
                else
                    model->meshes.push_back(mesh);
            }
        }

        std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
        return model;
    }

	void Model::Add(Model* model)
	{
		for (auto mesh : model->meshes) {
            meshes.push_back(mesh);
		}
	}


}
