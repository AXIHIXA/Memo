#version 450

layout (location = 0) out vec3 aColor;

vec2 pos[3] = vec2[](
    vec2( 0.0f, -0.5f),
    vec2( 0.5f,  0.5f),
    vec2(-0.5f,  0.5f)
);

vec3 color[3] = vec3[](
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f)
);

void main()
{
    gl_Position = vec4(pos[gl_VertexIndex], 0.0f, 1.0f);
    aColor = color[gl_VertexIndex];
}
