// Based on templates from learnopengl.com
// Authors: Matthew Lozito, Alex Kim
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

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
    
struct Sphere {
    glm::vec3 center;
    float radius;
    glm::vec3 color;
    
    Sphere(const glm::vec3& c, float r, const glm::vec3& col) 
        : center(c), radius(r), color(col) {}
};

struct Camera {
    glm::vec3 e, d, u, v, w;
    int width, height;
    float n , f;
    glm::vec3 topLeft;
    float pixelW, pixelH;
    Camera(const glm::vec3& origin, const glm::vec3& viewDir, const glm::vec3& up, int xRes, int yRes, float near, float far, float viewWidth, float viewHeight) 
        : e(origin), width(xRes), height(yRes), n(near), f(far) {
        d = glm::normalize(viewDir);
        v = glm::normalize(up);
        w = -d;
        u = glm::normalize(glm::cross(v, w));
        pixelW = viewWidth / float(width);
        pixelH = viewHeight / float(height);
        float halfW = 0.5f * viewWidth;
        float halfH = 0.5f * viewHeight;
        topLeft = (e + d * n) - u * halfW + v * halfH + u * (0.5f * pixelW) - v * (0.5f * pixelH);
    }
    glm::vec3 getPixelOrigin(int i, int j) const {
        return topLeft + u * (float(j) * pixelW) - v * (float(i) * pixelH);
    }
};

float raySphereIntersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const Sphere& sphere) {
    float a = glm::dot(rayDir, rayDir);
    glm::vec3 oc = rayOrigin - sphere.center;
    float tClosest = -glm::dot(oc, rayDir) / a;

    float r2 = sphere.radius * sphere.radius;
    float oc2 = glm::dot(oc, oc);
    if (tClosest < 0.0f && oc2 > r2) return -1.0f;

    float ocDotD = glm::dot(oc, rayDir);
    float b = 2.0f * ocDotD;
    float c = oc2 - r2;

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
    Camera cam = Camera(ORIGIN, X_AXIS, Z_AXIS, width, height, 0, 10, 10, 10);

    // Scene Objects
    Sphere s1 = Sphere({5, 2, 0}, 3, {255, 0, 0});
    Sphere s2 = Sphere({4, 1, 0}, 2, {0, 255, 0});
    Sphere s3 = Sphere({3, 0, 0}, 1, {0, 0, 255});

    std::vector<Sphere> sceneObjects = {s1, s2, s3};

    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            glm::vec3 curPixelOrigin = cam.getPixelOrigin(i, j);
            glm::vec3 curPixelRayDirection = cam.d;
            std::vector<float> results(sceneObjects.size(), -1.0f);
            for (int k = 0; k < sceneObjects.size(); k++) {
                results[k] = raySphereIntersect(curPixelOrigin, curPixelRayDirection, sceneObjects[k]);
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
            int idx = (i * width + j) * 3;
            if (closestIndex >= 0) {
                glm::vec3 color = sceneObjects[closestIndex].color;
                image[idx] = color.x; 
                image[idx+1] = color.y;
                image[idx+2] = color.z;
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