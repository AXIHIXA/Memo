#include <glm/glm.hpp>

#include "shape/Tetrahedron.h"
#include "util/Shader.h"


Tetrahedron::Tetrahedron(Shader * pShader, const glm::mat4 & model) : Mesh(pShader, model)
{
    // Initialize vertex data
    vertices.reserve(kNumberOfVertices);

    glm::vec3 normal314 = glm::normalize(glm::cross(V1 - V3, V4 - V1));
    vertices.emplace_back(V3, normal314, kColor);
    vertices.emplace_back(V1, normal314, kColor);
    vertices.emplace_back(V4, normal314, kColor);

    glm::vec3 normal321 = glm::normalize(glm::cross(V2 - V3, V1 - V2));
    vertices.emplace_back(V3, normal321, kColor);
    vertices.emplace_back(V2, normal321, kColor);
    vertices.emplace_back(V1, normal321, kColor);

    glm::vec3 normal124 = glm::normalize(glm::cross(V2 - V1, V4 - V2));
    vertices.emplace_back(V1, normal124, kColor);
    vertices.emplace_back(V2, normal124, kColor);
    vertices.emplace_back(V4, normal124, kColor);

    glm::vec3 normal342 = glm::normalize(glm::cross(V4 - V3, V2 - V4));
    vertices.emplace_back(V3, normal342, kColor);
    vertices.emplace_back(V4, normal342, kColor);
    vertices.emplace_back(V2, normal342, kColor);

    // OpenGL pipeline configuration
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizei>(this->vertices.size() * sizeof(Vertex)),
                 this->vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void Tetrahedron::render(float timeElapsedSinceLastFrame)
{
    pShader->use();
    pShader->setMat4("model", model);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glDrawArrays(GL_TRIANGLES,
                 0,                                       // start from index 0 in current VBO
                 static_cast<GLsizei>(vertices.size()));  // draw these number of elements

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}