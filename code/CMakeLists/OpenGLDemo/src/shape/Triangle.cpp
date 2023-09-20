#include <glm/glm.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>

#include "shape/Triangle.h"
#include "util/Shader.h"


Triangle::Triangle(Shader * pShader, const glm::mat3 & model) : Shape(pShader, model)
{
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Vertex coordinate attribute array "layout (position = 0) in vec2 aPosition"
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,                             // index: corresponds to "0" in "layout (position = 0)"
                          2,                             // size: each "vec2" generic vertex attribute has 2 values
                          GL_FLOAT,                      // data type: "vec2" generic vertex attributes are GL_FLOAT
                          GL_FALSE,                      // do not normalize data
                          sizeof(Vertex),                // stride between attributes in VBO data
                          reinterpret_cast<void *>(0));  // offset of 1st attribute in VBO data

    // Color vertex attribute array "layout (position = 1) in vec3 aColor"
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(Vertex),
                          reinterpret_cast<void *>(sizeof(Vertex::position)));

    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizei>(this->vertices.size() * sizeof(Vertex)),
                 this->vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


void Triangle::switchRotationStatus()
{
    rotationEnabled = !rotationEnabled;
}


void Triangle::render(float timeElapsedSinceLastFrame)
{
    if (rotationEnabled)
    {
        model = glm::rotate(model, timeElapsedSinceLastFrame);
    }

    pShader->use();
    pShader->setMat3("model", model);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glDrawArrays(GL_TRIANGLES,
                 0,                                       // start from index 0 in current VBO
                 static_cast<GLsizei>(vertices.size()));  // draw these number of elements

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
