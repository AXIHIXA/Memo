#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "shape/Shape.h"


class Shader;


class Triangle : private Shape
{
public:
    struct Vertex
    {
        glm::vec2 position;
        glm::vec3 color;
    };

    Triangle() = delete;
    Triangle(const Triangle &) = delete;
    Triangle & operator=(const Triangle &) = delete;

    Triangle(Shader * pShader, const glm::mat3 & model);

    Triangle(Triangle &&) noexcept = default;
    Triangle & operator=(Triangle &&) noexcept = default;
    ~Triangle() noexcept override = default;

    void switchRotationStatus();

    void render(float timeElapsedSinceLastFrame) override;

private:
    std::vector<Vertex> vertices
            {
                    // Vertex coordinate (screen-space coordinate), Vertex color
                    {{200.0f, 326.8f}, {1.0f, 0.0f, 0.0f}},
                    {{800.0f, 326.8f}, {0.0f, 1.0f, 0.0f}},
                    {{500.0f, 846.4f}, {0.0f, 0.0f, 1.0f}},
            };

    bool rotationEnabled {true};
};


#endif  // TRIANGLE_H
