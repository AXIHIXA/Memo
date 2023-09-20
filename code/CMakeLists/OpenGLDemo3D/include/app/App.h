#ifndef APP_H
#define APP_H

#include <memory>

#include <glm/glm.hpp>

#include "app/Window.h"
#include "util/Camera.h"


class Line;
class Mesh;
class Shader;
class Sphere;
class Tetrahedron;


class App : private Window
{
public:
    enum DisplayMode : int
    {
        kWireframe = 1,
        kFlat = 2,
        kNormal = 3,
        kSmooth = 4,
        kMaxDisplayModeEnum = 0x7FFFFFFF
    };

    static App & getInstance();

public:
    App(const App &) = delete;
    App(App &&) = delete;
    App & operator=(const App &) = delete;
    App & operator=(App &&) = delete;

    ~App() noexcept = default;

    void run();

private:
    static void cursorPosCallback(GLFWwindow *, double, double);
    static void framebufferSizeCallback(GLFWwindow *, int, int);
    static void keyCallback(GLFWwindow *, int, int, int, int);
    static void mouseButtonCallback(GLFWwindow *, int, int, int);
    static void scrollCallback(GLFWwindow *, double, double);

    static void perFrameTimeLogic(GLFWwindow *);
    static void processKeyInput(GLFWwindow *);

    // from CMakeLists.txt, compile definition
    static constexpr char kWindowName[] {WINDOW_NAME};
    static constexpr int kWindowWidth {1000};
    static constexpr int kWindowHeight {1000};

private:
    App();

    void initializeObjectsToRender();

    void render();

    // Objects to render
    std::unique_ptr<Shader> pLineShader;
    std::unique_ptr<Line> pAxes;

    std::unique_ptr<Shader> pMeshShader;
    std::unique_ptr<Tetrahedron> pTetrahedron;
    std::unique_ptr<Mesh> pTriangle;

    std::unique_ptr<Shader> pSphereShader;
    std::unique_ptr<Sphere> pSphere;

    // Viewing
    Camera camera {{0.0f, 0.0f, 10.0f}};
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    glm::vec3 lightColor {1.0f, 1.0f, 1.0f};
    glm::vec3 lightPos {10.0f, -10.0f, 10.0f};

    DisplayMode displayMode {kFlat};

    // Frontend GUI
    double timeElapsedSinceLastFrame {0.0};
    double lastFrameTimeStamp {0.0};

    bool mousePressed {false};
    glm::dvec2 mousePos {0.0, 0.0};

    // Used for camera movement from mouse dragging.
    // Note lastMouseLeftClickPos is different from lastMouseLeftPressPos.
    // If you press left button (and hold it there) and move the mouse,
    // lastMouseLeftPressPos gets updated to the current mouse position
    // (while lastMouseLeftClickPos, if there is one, remains the original value).
    glm::dvec2 lastMouseLeftClickPos {0.0, 0.0};
    glm::dvec2 lastMouseLeftPressPos {0.0, 0.0};
};


#endif  // APP_H
