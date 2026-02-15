// Based on templates from learnopengl.com
// Authors: Alex Kim, Matthew Lozito
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <type_traits>
#include <memory>
#include <cctype>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;


const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aColor;\n"
    "layout (location = 2) in vec2 aTexCoord;\n"
    "out vec3 ourColor;\n"
    "out vec2 TexCoord;\n"
    "void main()\n"
    "{\n"
	"gl_Position = vec4(aPos, 1.0);\n"
	"ourColor = aColor;\n"
	"TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
    "}\0";

const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec3 ourColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main()\n"
    "{\n"
    "   FragColor = texture(texture1, TexCoord);\n"
    "}\n\0";
    

class Shape {
    public:   
        virtual ~Shape() = default;
        virtual float getIntersection(const glm::vec3& p, const glm::vec3& d) const = 0;
        virtual glm::vec3 getColor() const = 0;
        virtual glm::vec3 getNormal(const glm::vec3& point) const = 0;
};

struct Sphere : public Shape {
public:
    glm::vec3 center;
    float radius;
    glm::vec3 color;
    
    Sphere(const glm::vec3 &c, float r, const glm::vec3 &col) 
        : center(c), radius(r), color(col) 
    {}
    
    float getIntersection(const glm::vec3& p, const glm::vec3& d) const override {
        float a = glm::dot(d, d);
    
        glm::vec3 sphereRay = p - center; // sphereRay is the ray from the ray origin to the center of the sphere
        float tClosest = -glm::dot(sphereRay, d) / a; // t value where the ray is closest to sphere center
    
        float r2 = radius * radius; 
        float sphereRay2 = glm::dot(sphereRay, sphereRay);
    
        // if the closest point is behind the ray origin and the origin is outside the sphere, we can't see the sphere
        if (tClosest < 0.0f && sphereRay2 > r2) return -1.0f; 
    
        float cDotD = glm::dot(sphereRay, d);
        float b = 2.0f * cDotD;
        float c = sphereRay2 - r2;
    
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) return -1.0f;
    
        float sqrtDisc = std::sqrt(disc);
    
        // Use the smaller root first
        float t0 = (-b - sqrtDisc) / (2.0f * a);
        float t1 = (-b + sqrtDisc) / (2.0f * a);
    
        if (t0 > 0) return t0;
        if (t1 > 0) return t1;
    
        return -1.0f;
    }

    glm::vec3 getColor() const override { return color; }
    glm::vec3 getNormal(const glm::vec3& point) const override {
        return glm::normalize(point - center);
    }
};   

class Triangle : public Shape {
public:
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
    glm::vec3 color;

    Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const glm::vec3 &color) 
        : a(a), b(b), c(c), color(color)
    {}

    float getIntersection(const glm::vec3& p, const glm::vec3& d) const override { 
        // small value to approximate 0
        const float EPS = 1e-7f;
    
        // vectors along a->b and a->c 
        glm::vec3 e1 = b - a;
        glm::vec3 e2 = c - a;
    
        glm::vec3 h = glm::cross(d, e2);
        float det = glm::dot(e1, h);
    
        // If det is near zero, ray is parallel to triangle plane
        if (fabs(det) < EPS) return -1.0f;
    
        float invDet = 1.0f / det;
    
        glm::vec3 s = p - a;
        float u = invDet * glm::dot(s, h); // Cramer's rule
        if (u < 0.0f || u > 1.0f) return -1.0f;
    
        glm::vec3 q = glm::cross(s, e1);
        float v = invDet * glm::dot(d, q); // Also Cramer's rule
        if (v < 0.0f || (u + v) > 1.0f) return -1.0f;
    
        float t = invDet * glm::dot(e2, q);
    
        if (t <= EPS) return -1.0f; // don't report hits from behind the camera
    
        return t;
    }

    glm::vec3 getColor() const override { return color; }
    glm::vec3 getNormal(const glm::vec3& point) const override {
        glm::vec3 ab = b - a;
        glm::vec3 ac = c - a;
        glm::vec3 normal = glm::cross(ab, ac);
        return glm::normalize(normal);
    }
};

class Plane : public Shape {
    public:
        glm::vec3 a;
        glm::vec3 n;
        glm::vec3 color;
    
        Plane(const glm::vec3 &a, const glm::vec3 &n, const glm::vec3 &color) 
            : a(a), n(n), color(color)
        {}
    
        float getIntersection(const glm::vec3& p, const glm::vec3& d) const override { 
            // small value to approximate 0
            const float EPS = 1e-7f;

            glm::vec3 nn = glm::normalize(n);
            float denom = glm::dot(nn, d);

            // No intersection if ray parallel to plane
            if (std::fabs(denom) < EPS) return -1.0f;

            float t = glm::dot(nn, (a - p)) / denom;

            // Intersection is behind the ray origin
            if (t <= EPS) return -1.0f;

            return t;
        }
    
        glm::vec3 getColor() const override { return color; }
        glm::vec3 getNormal(const glm::vec3& point) const override {
            return glm::normalize(n);
        }
    };

float k_a = 0.1;
float k_d = 0.3;
float k_s = 0.2;
float phongN = 2.0;

struct Light {
    glm::vec3 dir;
    glm::vec3 rayColor;
    float I;
    Light (const glm::vec3 &dir, const glm::vec3 &rayColor, float I)
        : dir(dir), rayColor(rayColor), I(I) {}
};

struct Camera {
    char mode;
    glm::vec3 e, d, u, v, w;
    int width, height;
    float n, f, p;
    float pixelW, pixelH, viewW, viewH;
    Camera(char cameraType, const glm::vec3& origin, const glm::vec3& viewDir, const glm::vec3& up, int xRes, int yRes, float near, float far, float viewWidth, float viewHeight, float perspectiveDistance) 
        : mode(cameraType), e(origin), width(xRes), height(yRes), viewW(viewWidth), viewH(viewHeight), n(near), f(far), p(perspectiveDistance) {
        d = glm::normalize(viewDir);
        v = glm::normalize(up);
        w = -d;
        u = glm::normalize(glm::cross(v, w));
        pixelW = viewW / float(width);
        pixelH = viewH / float(height);
    }

    std::pair<float, float> coordsToScreenSpace(int i, int j) const {
        float uPrime, vPrime;
        uPrime = -(0.5f * viewW) + pixelW * j + 0.5f * pixelW;
        vPrime = (0.5f * viewH) - pixelH * i - 0.5f * pixelH;
        return {uPrime, vPrime};
    }

    std::pair<glm::vec3, glm::vec3> getOrthographicRay(int i, int j) const {
        auto [uP, vP] = coordsToScreenSpace(i, j);
        glm::vec3 origin = e + u*uP + v*vP;
        glm::vec3 direction = d;
        return {origin, direction};
    }

    std::pair<glm::vec3, glm::vec3> getPerspectiveRay(int i, int j) const {
        auto [uP, vP] = coordsToScreenSpace(i, j);
        glm::vec3 origin = e;
        glm::vec3 direction = glm::normalize(d*p + u*uP + v*vP);
        return {origin, direction};
    }

    std::pair<glm::vec3, glm::vec3> getRay(int i, int j) const {
        if (tolower(mode) == 'o') {
            return getOrthographicRay(i, j);
        } else {
            return getPerspectiveRay(i, j);
        }
    }
};

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Display RGB Array", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // // GLEW: load all OpenGL function pointers
    glewInit();

    // build and compile the shaders
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {  
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // load and create a texture 
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // World Vectors
    const glm::vec3 ORIGIN{0.0f};
    const glm::vec3 X_AXIS{1.0f, 0.0f, 0.0f};
    const glm::vec3 Y_AXIS{0.0f, 1.0f, 0.0f};
    const glm::vec3 Z_AXIS{0.0f, 0.0f, 1.0f};    

    // Create the image (RGB Array) to be displayed
    const int width  = 512; // keep it in powers of 2!
    const int height = 512; // keep it in powers of 2!
    unsigned char image[width*height*3];

    // Camera Setup
    Camera cam = Camera('p', glm::vec3{0.0f, 3.0f, 0.0f}, glm::vec3{1.0f, -0.05f, 0.0f}, glm::vec3{0.05f, 1, 0.0f}, width, height, 0, 10, 10, 10, 5);

    // Scene Objects
    std::vector<std::unique_ptr<Shape>> sceneObjects;

    // Scene Lights
    std::vector<std::unique_ptr<Light>> sceneLights;

    //Light
    sceneLights.emplace_back(std::make_unique<Light>(
        glm::vec3{0.0f, -1.0f, 0.8f},    // direction
        glm::vec3{255.0f, 255.0f, 255.0f},    // ray color
        1.0f  // intensity
    ));
    //plane
    sceneObjects.emplace_back(std::make_unique<Plane>(
        glm::vec3{0.0f, 0.0f, 0.0f},    // point
        Y_AXIS,                         // normal
        glm::vec3{50.0f, 50.0f, 50.0f}  // color
    ));
    // sphere 1
    sceneObjects.emplace_back(std::make_unique<Sphere>(
        glm::vec3{15.0f, 4.0f, 4.0f},   // center
        4.0f,                           // radius
        glm::vec3{255.0f, 0.0f, 0.0f}   // color
    ));
    // sphere 2
    sceneObjects.emplace_back(std::make_unique<Sphere>(
        glm::vec3{10.0f, 2.0f, -3.0f},   // center
        2.0f,                           // radius
        glm::vec3{0.0f, 0.0f, 255.0f}   // color
    ));
    // triangle 1
    sceneObjects.emplace_back(std::make_unique<Triangle>(
        glm::vec3{8.0f, 4.0f, 2.0f},    // a
        glm::vec3{9.0f, 0.0f, -1.0f},   // b
        glm::vec3{7.0f, 0.0f, 1.0f},    // c
        glm::vec3{0.0f, 255.0f, 0.0f}   // color
    ));
    // triangle 2
    sceneObjects.emplace_back(std::make_unique<Triangle>(
        glm::vec3{8.0f, 4.0f, 2.0f},    // a
        glm::vec3{7.0f, 0.0f, 1.0f},    // b
        glm::vec3{9.0f, 0.0f, 4.0f},    // c
        glm::vec3{0.0f, 255.0f, 0.0f}   // color
    ));
    // triangle 3
    sceneObjects.emplace_back(std::make_unique<Triangle>(
        glm::vec3{8.0f, 4.0f, 2.0f},    // a
        glm::vec3{9.0f, 0.0f, 4.0f},    // b
        glm::vec3{9.0f, 0.0f, -1.0f},   // c
        glm::vec3{0.0f, 255.0f, 0.0f}   // color
    ));

    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            auto [curPixelOrigin, curPixelRayDirection] = cam.getRay(i, j);
            std::vector<float> results(sceneObjects.size(), -1);
            for (int k = 0; k < sceneObjects.size(); k++) {
                results[k] = sceneObjects[k]->getIntersection(curPixelOrigin, curPixelRayDirection);
            }
            float closestT = 1e9;
            int closestIndex = -1;
            for(int k = 0; k < results.size(); k++) {
                float curT = results[k];
                if (curT > 0 && curT < closestT) {
                    closestT = curT;
                    closestIndex = k;
                }
            }

            // glTexImage2D fills images bottom row to top, but we iterate top row to bottom
            int flippedI = (height - 1 - i); // so we need to flip our i to count down instead
            int idx = (flippedI * width + j) * 3;
            // only calculate color and lighting on ray hit
            if (closestIndex >= 0) {
                Shape* intersectedShape = sceneObjects[closestIndex].get();

                glm::vec3 intersectionPoint = curPixelOrigin + closestT * curPixelRayDirection;
                glm::vec3 normal = intersectedShape->getNormal(intersectionPoint);

                glm::vec3 LA = sceneObjects[closestIndex]->getColor() * k_a;
                glm::vec3 LDTot = {0.0f, 0.0f, 0.0f};
                glm::vec3 LSTot = {0.0f, 0.0f, 0.0f};

                for (int k = 0; k < sceneLights.size(); k++) {
                    bool shadow = false;
                    glm::vec3 pointToLightDir = -(sceneLights[k]->dir);
                    for (int l = 0; l < sceneObjects.size(); l++) {
                        float hit = sceneObjects[l]->getIntersection(intersectionPoint, pointToLightDir);
                        float EPS = 1e-3;
                        if (hit > EPS) {
                            shadow = true; 
                            break;
                        } 
                    }
                    if (!shadow) {
                        glm::vec3 LD = k_d * sceneLights[k]->I * sceneObjects[closestIndex]->getColor() * std::max(0.0f, glm::dot(normal, -sceneLights[k]->dir));
                        LDTot += LD;
                        glm::vec3 VR = 2 * glm::dot(normal, -sceneLights[k]->dir) * normal + sceneLights[k]->dir;
                        glm::vec3 LS = k_s * sceneLights[k]->I * sceneLights[k]->rayColor * std::pow(std::max(0.0f, glm::dot(-curPixelRayDirection, VR)), phongN);
                        LSTot += LS;
                    }
                }
                glm::vec3 L = LA + LDTot + LSTot;
                L = glm::clamp(L, 0.0f, 255.0f);
                image[idx] = L.x;
                image[idx+1] = L.y;
                image[idx+2] = L.z;
            } else {
                image[idx] = 0; 
                image[idx+1] = 0;
                image[idx+2] = 0;
            }

            
        }
    }

    unsigned char *data = &image[0];
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
   


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind Texture
        glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}