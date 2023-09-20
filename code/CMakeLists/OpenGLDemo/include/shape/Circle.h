#ifndef CIRCLE_H
#define CIRCLE_H


#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>


class Shader;


// Circle[s] class, this class represents MULTIPLE circles.
class Circle
{
public:
    Circle() = delete;

    explicit Circle(Shader * pShader);

    Circle(const Circle &) = delete;
    Circle & operator=(const Circle &) = delete;

    Circle(Circle &&) noexcept;
    Circle & operator=(Circle &&) noexcept;

    ~Circle() noexcept;

    void render();

private:
    Shader * pShader {nullptr};

    std::vector<glm::vec3> parameters
            {
                    // Coordinate (x, y) of the center, radius (all in NDC)
                    {0.0f, 0.0f, 0.7f},
                    {-0.8f, -0.8f, 0.1f}
            };

    GLuint vao {0};
    GLuint vbo {0};
};


#endif  // CIRCLE_H
