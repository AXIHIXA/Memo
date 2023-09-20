#include "shape/Circle.h"
#include "util/Shader.h"


Circle::Circle(Shader * pShader) : pShader(pShader)
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Vertex coordinate attribute array "layout (position = 0) in vec3 aPos"
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,                             // index: corresponds to "0" in "layout (position = 0)"
                          3,                             // size: each "vec3" generic vertex attribute has 3 values
                          GL_FLOAT,                      // data type: "vec3" generic vertex attributes are GL_FLOAT
                          GL_FALSE,                      // do not normalize data
                          sizeof(glm::vec3),             // stride between attributes in VBO data
                          reinterpret_cast<void *>(0));  // offset of 1st attribute in VBO data

    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizei>(parameters.size() * sizeof(glm::vec3)),
                 parameters.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


Circle::Circle(Circle && rhs) noexcept
{
    *this = std::move(rhs);
}


Circle & Circle::operator=(Circle && rhs) noexcept
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

    parameters = std::move(rhs.parameters);

    return *this;
}


Circle::~Circle() noexcept
{
    glDeleteVertexArrays(1, &vao);
    vao = 0U;

    glDeleteBuffers(1, &vbo);
    vbo = 0U;
}


void Circle::render()
{
    pShader->use();

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glDrawArrays(GL_PATCHES,
                 0,                                          // start from index 0 in current VBO
                 static_cast<GLsizei>(parameters.size()));  // draw these number of elements

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
