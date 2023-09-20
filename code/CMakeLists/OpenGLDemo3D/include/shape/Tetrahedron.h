#ifndef TETRAHEDRON_H
#define TETRAHEDRON_H

#include <glm/glm.hpp>

#include "shape/Mesh.h"


class Shader;


class Tetrahedron : private Mesh
{
public:
    Tetrahedron() = delete;
    Tetrahedron(const Tetrahedron &) = delete;
    Tetrahedron & operator=(const Tetrahedron &) = delete;

    Tetrahedron(Shader * pShader, const glm::mat4 & model);

    Tetrahedron(Tetrahedron &&) noexcept = default;
    Tetrahedron & operator=(Tetrahedron &&) noexcept = default;
    ~Tetrahedron() noexcept override = default;

    void render(float timeElapsedSinceLastFrame) override;

private:
    static constexpr std::size_t kNumberOfFacets {4UL};
    static constexpr std::size_t kNumberOfVertices {3UL * kNumberOfFacets};

    static constexpr GLfloat kRadius {0.5773502691896258f};
    static constexpr glm::vec3 V1 {kRadius, kRadius, kRadius};
    static constexpr glm::vec3 V2 {-kRadius, -kRadius, kRadius};
    static constexpr glm::vec3 V3 {-kRadius, kRadius, -kRadius};
    static constexpr glm::vec3 V4 {kRadius, -kRadius, -kRadius};

    static constexpr glm::vec3 kColor {0.31f, 0.5f, 1.0f};
};


#endif  // TETRAHEDRON_H
