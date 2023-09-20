#ifndef SHAPE_H
#define SHAPE_H

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>


class Shader;


/// Generic Shape object that manages OpenGL objects for a shape.
/// All shapes are designed to private- or protected-inherit the Shape class.
class Shape
{
public:
    Shape() = delete;
    Shape(const Shape &) = delete;
    Shape & operator=(const Shape &) = delete;

    virtual ~Shape() noexcept = 0;

    virtual void render(float timeSinceLastFrame) = 0;

protected:
    Shape(Shader * pShader, const glm::mat4 & model);

    Shape(Shape &&) noexcept;
    Shape & operator=(Shape &&) noexcept;

    Shader * pShader {nullptr};

    GLuint vao {0U};
    GLuint vbo {0U};

    glm::mat4 model {glm::mat4(1.0f)};
};


#endif  // SHAPE_H
