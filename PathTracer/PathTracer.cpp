#include "PathTracer.h"
#include "Renderer.h"
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

namespace pt {

    struct SampleWindow : public osc::GLFCameraWindow
    {
        SampleWindow(const std::string& title,
            const Model* model,
            const Camera& camera,
            const float worldScale)
            : GLFCameraWindow(title, camera.pos, camera.at, camera.up, worldScale),
            sample(model)
        {
            sample.setCamera(camera);
        }

        virtual void render() override
        {
            if (cameraFrame.modified) {
                sample.setCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }
            sample.render();
        }

        virtual void draw() override
        {
            sample.downloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_FLOAT;//GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        virtual void resize(const vec2i& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }

        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        Renderer        sample;
        std::vector<vec4f> pixels;
    };

	extern "C" int main()
	{
		std::string inputfile("../../../models/CornellBox1/CornellBox-Original.obj");
		Model* scene = loadOBJ(inputfile);
        Model* bunny = loadOBJ("../../../models/bunny/bunny.obj");
        //scene->Add(bunny);
		Camera camera{
			vec3f(0.f, 1.f, 3.f), // pos
			vec3f(0.f, 1.f, -1.f), // at
			vec3f(0.f, 1.f, 0.f) // up
		}; // from (0,0,0) looking at (0,0,-1)
		Renderer render(scene);
		render.setCamera(camera);
        vec2i fbSize(1200, 1024);
        render.resize(fbSize);
        render.render();
        std::vector<uint32_t> pixels(fbSize.x, fbSize.y);

        SampleWindow* window = new SampleWindow("Optix 7",
            scene, camera, 1.f);

        window->run();
        return 0;
	}

}
