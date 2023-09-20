#include "shape/Shape.h"


Shape::~Shape() noexcept
{
    glDeleteVertexArrays(1, &vao);
    vao = 0U;

    glDeleteBuffers(1, &vbo);
    vbo = 0U;
}


Shape::Shape(Shader * pShader, const glm::mat3 & model) : pShader(pShader), model(model)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
}


Shape::Shape(Shape && rhs) noexcept
{
    *this = std::move(rhs);
}


Shape & Shape::operator=(Shape && rhs) noexcept
{
    if (this == &rhs)
    {
        return *this;
    }

    pShader = rhs.pShader;
    rhs.pShader = nullptr;

    vao = rhs.vao;
    rhs.vao = 0U;

    vbo = rhs.vbo;
    rhs.vbo = 0U;

    model = rhs.model;

    return *this;
}
