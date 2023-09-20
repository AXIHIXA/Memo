#ifndef VULKANAPP_H
#define VULKANAPP_H

#include <optional>
#include <string_view>
#include <vector>

#include <GLFW/glfw3.h>


class VulkanApp
{
public:
    VulkanApp() = delete;
    VulkanApp(std::string_view vert, std::string_view frag);
    VulkanApp(const VulkanApp &) = delete;
    VulkanApp(VulkanApp &&) = delete;
    VulkanApp & operator=(const VulkanApp &) = delete;
    VulkanApp & operator=(VulkanApp &&) = delete;
    ~VulkanApp();

    // Run this app, the only public interface needed
    void run();

private:
    // typedefs, static & const members
    struct QueueFamilyIndex
    {
        [[nodiscard]] bool isComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }

        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
    };

    struct SwapChainSupport
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    // GLFW GUI callbacks (static part).
    // GLFW does not know C++ member functions (i.e., `this` pointer).
    // Doc says we should pass in regular functions or static member functions.
    static void cursorPosCallback(GLFWwindow * window, double, double);
    static void framebufferSizeCallback(GLFWwindow *, int, int);
    static void keyCallback(GLFWwindow *, int, int, int, int);
    static void mouseButtonCallback(GLFWwindow *, int, int, int);
    static void scrollCallback(GLFWwindow *, double, double);

    // Vulkan validation layer diagnostics callback.
    // Add a new static member function called debugCallback
    // with the PFN_vkDebugUtilsMessengerCallbackEXT prototype.
    // The VKAPI_ATTR and VKAPI_CALL ensure that
    // the function has the right signature for Vulkan to call it.
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
            void * pUserData);

    // Static utilities
    static std::vector<char> read(std::string_view name);

    // Constants
    static constexpr char kWindowName[] {WINDOW_NAME};
    static constexpr int kWindowWidth {1000};
    static constexpr int kWindowHeight {1000};

    // Used for `pName` field for VkPipelineShaderStageCreateInfo
    static constexpr const char * const kGlslOpEntryPoint {"main"};

    // Max number of frames being rendered simultaneously
    static constexpr std::size_t kMaxFramesInFlight {2U};

    // Vulkan configurations
#ifdef NDEBUG
    const bool enableValidationLayers {false};
#else
    const bool enableValidationLayers {true};
#endif  // NDEBUG

    const std::vector<const char *> requiredValidationLayers {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> requiredDeviceExtensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Viewport(s) and scissor rectangle(s) can either be specified
    // as a static part of the graphicsPipeline or as a dynamic state set in the command buffer.
    // While the former is more in line with the other states,
    // it's often convenient to make viewport and scissor state dynamic
    // as it gives you a lot more flexibility.
    // This is very common and all implementations can handle this dynamic state
    // without a performance penalty.
    const std::vector<VkDynamicState> dynamicStates {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

private:
    // non-static non-const memebrs
    // Initialization/destruction helpers
    void initGLFW();

    void initVulkan(std::string_view vert, std::string_view frag);

    void createInstance();
    bool supportsAllRequiredValidationLayers();
    [[nodiscard]] std::vector<const char *> getRequiredExtensions() const;
    VkResult
    createDebugUtilsMessengerEXT(
            const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
            const VkAllocationCallbacks * pAllocator,
            VkDebugUtilsMessengerEXT * pDebugMessenger);
    void destroyDebugUtilsMessengerEXT(const VkAllocationCallbacks * pAllocator);

    void createWindowSurface();

    void createDevice();
    QueueFamilyIndex findQueueFamilies(VkPhysicalDevice physicalDevice);
    bool isSuitablePhysicalDevice(VkPhysicalDevice physicalDevice);
    bool supportsAllReuiredDeviceExtensions(VkPhysicalDevice physicalDevice);

    void createSwapChain();
    SwapChainSupport querySwapChainSupport(VkPhysicalDevice physicalDevice);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createImageViews();

    void createGraphicsPipeline(std::string_view vert, std::string_view frag);
    void createRenderPass();
    VkShaderModule createShaderModule(const std::vector<char> & code);

    void createFramebuffers();

    void createCommandPool();

    void createCommandBuffers();
    void recordCommandBuffer(VkCommandBuffer commandBuffer_, uint32_t imageIndex);

    void createSyncObjects();

    void glfwCleanup();
    void vulkanCleanup();

    // Vulkan render-time utilities
    void drawFrame(std::size_t frameIndex);

    void recreateSwapChain();
    void cleanupSwapChain();

    // GLFW GUI callbacks (member part)
    void perFrameTimeLogic();
    void processKeyInput();

    // GLFW window and GUI status variables
    GLFWwindow * pWindow {nullptr};

    double timeSinceLastFrame {0.0};
    double lastFrameTimeStamp {0.0};

    bool firstMouse {true};
    bool mousePressed {false};

    double mouseXPos {0.0};
    double mouseYPos {0.0};

    double lastMousePressXPos {0.0};
    double lastMousePressYPos {0.0};

    bool framebufferResized {false};

    // Vulkan objects
    VkInstance instance {nullptr};
    VkDebugUtilsMessengerEXT debugUtilsMessenger {nullptr};

    VkSurfaceKHR surface {nullptr};

    VkPhysicalDevice physicalDevice {VK_NULL_HANDLE};
    VkDevice device {nullptr};
    VkQueue graphicsQueue {nullptr};
    VkQueue presentQueue {nullptr};

    VkSwapchainKHR swapChain {nullptr};
    std::vector<VkImage> swapChainImages {};
    VkFormat swapChainImageFormat {};
    VkExtent2D swapChainExtent {};
    std::vector<VkImageView> swapChainImageViews {};

    VkPipeline graphicsPipeline {nullptr};
    VkPipelineLayout pipelineLayout {nullptr};
    VkRenderPass renderPass {nullptr};

    std::vector<VkFramebuffer> swapChainFramebuffers {};

    VkCommandPool commandPool {nullptr};

    // Per-frame attributes.
    // Each frame should have its own command buffer, set of semaphores, and fence.
    std::vector<VkCommandBuffer> commandBuffers {};

    std::vector<VkSemaphore> imageAvailableSemaphores {};
    std::vector<VkSemaphore> renderFinishedSemaphores {};
    std::vector<VkFence> inFlightFences {};
};


#endif  // VULKANAPP_H
