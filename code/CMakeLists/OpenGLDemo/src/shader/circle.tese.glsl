#version 410 core

layout (isolines, equal_spacing, ccw) in;

const float kPi = 3.14159265358979323846f;

void main()
{
    vec4 params = gl_in[0].gl_Position;
    vec4 c = vec4(params.xy, 0, 0);
    float r = params.z;

    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // Use u here because the isolines mode only connects mesh grids horizontally!
    // If we use v here, theta only differs vertically and cooresponding vertices will not be connected!
    float theta = 2 * kPi * u;

    // Try the following and see the difference:
    // 1. Use v in theta and there will be NO output;
    // 2. Add "point_mode" to line 3 as "layout (isolines, equal_spacing, ccw, point_mode) in";
    // 3. Run the same program again and see the dot output!

    gl_Position = vec4(r * cos(theta), r * sin(theta), 0, 1) + c;
}
