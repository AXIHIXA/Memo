#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fmt/core.h>

#include "app/VulkanApp.h"


VulkanApp::VulkanApp(std::string_view vert, std::string_view frag)
{
    initGLFW();
    initVulkan(vert, frag);
}


VulkanApp::~VulkanApp()
{
    vulkanCleanup();
    glfwCleanup();
}


void VulkanApp::run()
{
    for (std::size_t frameIndex = 0U;
         !glfwWindowShouldClose(pWindow);
         frameIndex = (frameIndex + 1U) % kMaxFramesInFlight)
    {
        perFrameTimeLogic();
        processKeyInput();

        drawFrame(frameIndex);

        glfwSwapBuffers(pWindow);
        glfwPollEvents();
    }

    vkDeviceWaitIdle(device);
}


void VulkanApp::cursorPosCallback(GLFWwindow * window, double xpos, double ypos)
{
    auto pContext = static_cast<VulkanApp *>(glfwGetWindowUserPointer(window));

    pContext->mouseXPos = xpos;
    pContext->mouseYPos = VulkanApp::kWindowHeight - ypos;

    if (pContext->firstMouse)
    {
        pContext->lastMousePressXPos = pContext->mouseXPos;
        pContext->lastMousePressYPos = pContext->mouseYPos;
        pContext->firstMouse = false;
    }

//    double xoffset = context.mouseXPos - context.lastMousePressXPos;
//    double yoffset = context.mouseYPos - context.lastMousePressYPos;

    pContext->lastMousePressXPos = pContext->mouseXPos;
    pContext->lastMousePressYPos = pContext->mouseYPos;
}


void VulkanApp::framebufferSizeCallback(GLFWwindow * window, int width, int height)
{
    auto pContext = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));
    pContext->framebufferResized = true;
}


void VulkanApp::keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{

}


void VulkanApp::mouseButtonCallback(GLFWwindow * window, int button, int action, int mods)
{
    auto pContext = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        pContext->mousePressed = true;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        pContext->mousePressed = false;
    }
}


void VulkanApp::scrollCallback(GLFWwindow * window, double xoffset, double yoffset)
{

}


namespace
{

std::string toString(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity)
{
    switch (messageSeverity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        return "VERBOSE";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        return "INFO";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        return "WARNING";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        return "ERROR";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
    default:
        return "UNKNOWN";
    }
}


std::string toString(VkDebugUtilsMessageTypeFlagsEXT messageType)
{
    switch (messageType)
    {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
        return "GENERAL";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
        return "VALIDATION";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
        return "PERFORMANCE";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
        return "DEVICE_ADDRESS_BINDING";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_FLAG_BITS_MAX_ENUM_EXT:
    default:
        return "UNKNOWN";
    }
}

}  // namespace anamynous


VKAPI_ATTR VkBool32 VKAPI_CALL
VulkanApp::debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
        void * pUserData)
{
    fmt::print(stdout,
               "VkDebugUtils [{:>7}] [{:>11}] - {}\n",
               toString(messageSeverity),
               toString(messageType),
               pCallbackData->pMessage);

    return VK_FALSE;
}


std::vector<char> VulkanApp::read(std::string_view name)
{
    //    ate: Start reading at the end of the file;
    // binary: Read the file as binary file (avoid text transformations).
    if (std::ifstream fin {name.data(), std::ios::ate | std::ios::binary})
    {
        auto fileSize {fin.tellg()};
        std::vector<char> buffer(fileSize);
        fin.seekg(0);
        fin.read(buffer.data(), fileSize);
        return buffer;
    }
    else
    {
        throw std::runtime_error(fmt::format("{} {}", "failed to open", name));
    }
}


void VulkanApp::initGLFW()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    pWindow = glfwCreateWindow(VulkanApp::kWindowWidth,
                               VulkanApp::kWindowHeight,
                               VulkanApp::kWindowName,
                               nullptr,
                               nullptr);

    if (!pWindow)
    {
        throw std::runtime_error("failed to create GLFW pWindow");
    }

    // GLFW Offical FAQ:
    // Store the pointer to the object you wish to call
    // as the user pointer for the window
    // and use it to call methods on your object.
    // https://www.glfw.org/faq.html#216---how-do-i-use-c-methods-as-callbacks
    // https://stackoverflow.com/questions/7676971/pointing-to-a-function-that-is-a-class-member-glfw-setkeycallback
    glfwMakeContextCurrent(pWindow);
    glfwSetWindowUserPointer(pWindow, this);
    glfwSetCursorPosCallback(pWindow, cursorPosCallback);
    glfwSetFramebufferSizeCallback(pWindow, framebufferSizeCallback);
    glfwSetKeyCallback(pWindow, keyCallback);
    glfwSetMouseButtonCallback(pWindow, mouseButtonCallback);
    glfwSetScrollCallback(pWindow, scrollCallback);
}


void VulkanApp::initVulkan(std::string_view vert, std::string_view frag)
{
    createInstance();
    createWindowSurface();
    createDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline(vert, frag);
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}


void VulkanApp::createInstance()
{
    // 1. Create Vulkan instance

    // For vkValidationLayer logging callback messenger creation.
    // Put ahead to pass into vkInstance create info.
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo
        {
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                nullptr,
                0U,
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                debugCallback,
                nullptr
        };

    // App info
    VkApplicationInfo appInfo
        {
                VK_STRUCTURE_TYPE_APPLICATION_INFO,
                nullptr,
                "Vulkan App",
                VK_MAKE_VERSION(1, 0, 0),
                "No Engine",
                VK_MAKE_VERSION(1, 0, 0),
                VK_API_VERSION_1_0
        };

    // Check Vulkan validation layer support
    if (enableValidationLayers && !supportsAllRequiredValidationLayers())
    {
        throw std::runtime_error("validation layers requested but not available");
    }

    // GLFW Extensions
    std::vector<const char *> extensions = getRequiredExtensions();

    // Create info (as argument to vkCreateInstance)
    VkInstanceCreateInfo instanceCreateInfo
            {
                    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    enableValidationLayers ? &debugUtilsMessengerCreateInfo : nullptr,
                    0U,
                    &appInfo,
                    enableValidationLayers ? static_cast<uint32_t>(requiredValidationLayers.size()) : 0U,
                    enableValidationLayers ? requiredValidationLayers.data() : nullptr,
                    static_cast<uint32_t>(extensions.size()),
                    extensions.data()
            };

    // Create Vulkan instance
    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create Vulkan instance");
    }

    // 2. Setup Vulkan debug utils messenger
    if (enableValidationLayers)
    {
        if (createDebugUtilsMessengerEXT(
                &debugUtilsMessengerCreateInfo,
                nullptr,
                &debugUtilsMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to setup debug messenger");
        }
    }
}


bool VulkanApp::supportsAllRequiredValidationLayers()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const auto & layerName : requiredValidationLayers)
    {
        if (std::none_of(availableLayers.cbegin(),
                         availableLayers.cend(),
                         [layerName](const auto & layerProperty)
                         {
                             return !std::strcmp(layerName, layerProperty.layerName);
                         }))
        {
            return false;
        }
    }

    return true;
}


std::vector<const char *> VulkanApp::getRequiredExtensions() const
{
    uint32_t glfwExtensionCount = 0;
    const char ** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}


VkResult
VulkanApp::createDebugUtilsMessengerEXT(
        const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
        const VkAllocationCallbacks * pAllocator,
        VkDebugUtilsMessengerEXT * pDebugMessenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

    return func ? func(instance, pCreateInfo, pAllocator, pDebugMessenger) : VK_ERROR_EXTENSION_NOT_PRESENT;
}


void VulkanApp::destroyDebugUtilsMessengerEXT(const VkAllocationCallbacks* pAllocator)
{
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    if (func)
    {
        func(instance, debugUtilsMessenger, pAllocator);
    }
}


void VulkanApp::createWindowSurface()
{
    // Create window surface
    if (glfwCreateWindowSurface(instance, pWindow, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface");
    }
}


void VulkanApp::createDevice()
{
    // 1. Pick a suitable physical device
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (!deviceCount)
    {
        throw std::runtime_error("failed to find physical devices with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    auto itPhysicalDevice =
            std::find_if(devices.cbegin(),
                         devices.cend(),
                         [this](const auto & d)
                         {
                             return isSuitablePhysicalDevice(d);
                         });

    if (itPhysicalDevice == devices.end())
    {
        throw std::runtime_error("failed to find a suitable physical device");
    }
    else
    {
        physicalDevice = *itPhysicalDevice;
    }

    // 2. Create the Vulkan logical device

    QueueFamilyIndex index = findQueueFamilies(physicalDevice);
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo deviceQueueCreateInfo
            {
                    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    nullptr,
                    0U,
                    index.graphicsFamily.value(),
                    1,
                    &queuePriority
            };

    // Right now we don't need anything special, so we can simply define itPhysicalDevice and leave everything to VK_FALSE.
    // We'll come back to this structure once we're about to start doing more interesting things with Vulkan.
    VkPhysicalDeviceFeatures physicalDeviceFeatures {};

    VkDeviceCreateInfo deviceCreateInfo
            {
                    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                    nullptr,
                    0U,
                    1,
                    &deviceQueueCreateInfo,
                    enableValidationLayers ? static_cast<uint32_t>(requiredValidationLayers.size()) : 0,
                    enableValidationLayers ? requiredValidationLayers.data() : nullptr,
                    static_cast<uint32_t>(requiredDeviceExtensions.size()),
                    requiredDeviceExtensions.data(),
                    &physicalDeviceFeatures
            };

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device");
    }

    // 3. Get device queues.
    // The queues are automatically created along with the logical device,
    // but we don't have a handle to interface with them yet.
    // Device queues are implicitly cleaned up when the device is destroyed.
    vkGetDeviceQueue(device, index.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, index.presentFamily.value(), 0, &presentQueue);
}


VulkanApp::QueueFamilyIndex VulkanApp::findQueueFamilies(VkPhysicalDevice phyDevice)
{
    // Return value
    QueueFamilyIndex index;

    // Get all available queue families on the given physicalDevice
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice,
                                             &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice,
                                             &queueFamilyCount,
                                             queueFamilies.data());

    // Find a suitable graphics queue (that accepts drawing commands)
    // and a suitable present queue (that shows/presents rendered image to window surface)
    VkBool32 supportsPresent {VK_FALSE};

    for (std::uint32_t i = 0; i != queueFamilies.size() && !index.isComplete(); ++i)
    {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            index.graphicsFamily = i;
        }

        vkGetPhysicalDeviceSurfaceSupportKHR(phyDevice,
                                             i,
                                             surface,
                                             &supportsPresent);

        if (supportsPresent)
        {
            index.presentFamily = i;
        }
    }

    return index;
}


bool VulkanApp::isSuitablePhysicalDevice(VkPhysicalDevice phyDevice)
{
    QueueFamilyIndex index = findQueueFamilies(phyDevice);

    bool extensionsSupported = supportsAllReuiredDeviceExtensions(phyDevice);

    bool swapChainAdequate;

    if (extensionsSupported)
    {
        SwapChainSupport swapChainSupport = querySwapChainSupport(phyDevice);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return index.isComplete() && extensionsSupported && swapChainAdequate;
}


bool VulkanApp::supportsAllReuiredDeviceExtensions(VkPhysicalDevice phyDevice)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(phyDevice,
                                         nullptr,
                                         &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(phyDevice,
                                         nullptr,
                                         &extensionCount,
                                         availableExtensions.data());

    for (const auto & deviceExtension : requiredDeviceExtensions)
    {
        if (std::none_of(availableExtensions.cbegin(),
                         availableExtensions.cend(),
                         [deviceExtension](const auto & extensionProperty)
                         {
                             return !std::strcmp(deviceExtension, extensionProperty.extensionName);
                         }))
        {
            return false;
        }
    }

    return true;
}


void VulkanApp::createSwapChain()
{
    SwapChainSupport swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (0 < swapChainSupport.capabilities.maxImageCount && swapChainSupport.capabilities.maxImageCount < imageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    QueueFamilyIndex index = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {index.graphicsFamily.value(), index.presentFamily.value()};

    VkSwapchainCreateInfoKHR createInfo
            {
                    VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                    nullptr,
                    0U,
                    surface,
                    imageCount,
                    surfaceFormat.format,
                    surfaceFormat.colorSpace,
                    extent,
                    1U,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    index.graphicsFamily == index.presentFamily ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
                    index.graphicsFamily == index.presentFamily ? 0U : 2U,
                    index.graphicsFamily == index.presentFamily ? nullptr : queueFamilyIndices,
                    swapChainSupport.capabilities.currentTransform,
                    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                    presentMode,
                    VK_TRUE,
                    VK_NULL_HANDLE
            };

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create swap chain");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}


VulkanApp::SwapChainSupport VulkanApp::querySwapChainSupport(VkPhysicalDevice phyDevice)
{
    SwapChainSupport details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phyDevice,
                                              surface,
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice,
                                         surface,
                                         &formatCount,
                                         nullptr);

    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phyDevice,
                                             surface,
                                             &formatCount,
                                             details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice,
                                              surface,
                                              &presentModeCount,
                                              nullptr);

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(phyDevice,
                                                  surface,
                                                  &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}


#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"

VkSurfaceFormatKHR VulkanApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & availableFormats)
{
    for (const auto & availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}


VkPresentModeKHR VulkanApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> & availablePresentModes)
{
    for (const auto & availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

#pragma clang diagnostic pop


VkExtent2D VulkanApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR & capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(pWindow, &width, &height);

        VkExtent2D actualExtent
                {
                        static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height)
                };

        actualExtent.width = std::clamp(actualExtent.width,
                                        capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,
                                         capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);

        return actualExtent;
    }
}


void VulkanApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (std::size_t i = 0; i != swapChainImages.size(); ++i)
    {
        VkImageViewCreateInfo createInfo
                {
                        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        nullptr,
                        0U,
                        swapChainImages[i],
                        VK_IMAGE_VIEW_TYPE_2D,
                        swapChainImageFormat,
                        {VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY},
                        {VK_IMAGE_ASPECT_COLOR_BIT,
                         0U,
                         1U,
                         0U,
                         1U}
                };

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i])
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image view #" + std::to_string(i));
        }
    }
}


void VulkanApp::createGraphicsPipeline(std::string_view vert, std::string_view frag)
{
    VkShaderModule vertShaderModule = createShaderModule(read(vert));
    VkShaderModule fragShaderModule = createShaderModule(read(frag));

    VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    vertShaderModule,
                    kGlslOpEntryPoint,
                    nullptr  // specify values for shader constants here
            };

    VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    fragShaderModule,
                    kGlslOpEntryPoint,
                    nullptr  // specify values for shader constants here
            };

    VkPipelineShaderStageCreateInfo shaderStages[] {vertShaderStageCreateInfo, fragShaderStageCreateInfo};

    // Because we're hard coding the vertex data directly in the vertex shader,
    // we'll fill in this structure to specify that there is no vertex data to load for now.
    // We'll get back to it in the vertex buffer chapter.
    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    0U,
                    nullptr,
                    0U,
                    nullptr
            };

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    VK_FALSE
            };

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    1U,
                    nullptr,  // dynamic state, set at runtime
                    1U,
                    nullptr  // dynamic state, set at runtime
            };

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_FALSE,
                    VK_FALSE,
                    VK_POLYGON_MODE_FILL,
                    VK_CULL_MODE_BACK_BIT,
                    VK_FRONT_FACE_CLOCKWISE,
                    VK_FALSE,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f
            };

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_SAMPLE_COUNT_1_BIT,
                    VK_FALSE,
                    0.0f,
                    nullptr,
                    VK_FALSE,
                    VK_FALSE
            };

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState
            {
                    VK_FALSE,
                    VK_BLEND_FACTOR_ZERO,
                    VK_BLEND_FACTOR_ZERO,
                    VK_BLEND_OP_ADD,
                    VK_BLEND_FACTOR_ZERO,
                    VK_BLEND_FACTOR_ZERO,
                    VK_BLEND_OP_ADD,
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            };

    VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    VK_FALSE,
                    VK_LOGIC_OP_COPY,
                    1U,
                    &colorBlendAttachmentState,
                    {0.0f, 0.0f, 0.0f, 0.0f}
            };

    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                    nullptr,
                    0U,
                    static_cast<uint32_t>(dynamicStates.size()),
                    dynamicStates.data()
            };

    // Shader uniforms need to be specified during graphicsPipeline creation by creating a VkPipelineLayout object.
    // The structure also specifies push constants,
    // which are another way of passing dynamic values to shaders.
    // The graphicsPipeline layout will be referenced throughout the program's lifetime.
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo
            {
                    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    nullptr,
                    0U,
                    0U,
                    nullptr,
                    0U,
                    nullptr
            };

    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout)
        != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphicsPipeline layout");
    }

    // We can now combine all the structures and objects to create the graphics pipeline!
    // Here's the types of objects we have now, as a quick recap:
    //            Shader stages: The shader modules that define the functionality
    //                           of the programmable stages of the graphics pipeline;
    //     Fixed-function state: All the structures that define the fixed-function stages of the pipeline,
    //                           like input assembly, rasterizationStateCreateInfo, viewport and color blending;
    //          Pipeline layout: The uniform and push values referenced by the shader
    //                           that can be updated at drawFrame time;
    //              Render pass: The attachments referenced by the pipeline stages and their usage.
    VkGraphicsPipelineCreateInfo pipelineInfo
            {
                    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    nullptr,
                    0U,
                    2U,
                    shaderStages,
                    &vertexInputStateCreateInfo,
                    &inputAssemblyStateCreateInfo,
                    nullptr,
                    &viewportStateCreateInfo,
                    &rasterizationStateCreateInfo,
                    &multisampleStateCreateInfo,
                    nullptr,
                    &colorBlendStateCreateInfo,
                    &dynamicStateCreateInfo,
                    pipelineLayout,
                    renderPass,
                    0U,
                    VK_NULL_HANDLE,
                    -1
            };

    if (vkCreateGraphicsPipelines(device,
                                  VK_NULL_HANDLE,
                                  1,
                                  &pipelineInfo,
                                  nullptr,
                                  &graphicsPipeline)
        != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline");
    }

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}


void VulkanApp::createRenderPass()
{
    VkAttachmentDescription colorAttachment
            {
                    0U,
                    swapChainImageFormat,
                    VK_SAMPLE_COUNT_1_BIT,
                    VK_ATTACHMENT_LOAD_OP_CLEAR,
                    VK_ATTACHMENT_STORE_OP_STORE,
                    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            };

    VkAttachmentReference colorAttachmentRef
            {
                    0U,  // which attachment to reference by its index in the attachment array
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            };

    // The following other types of attachments can be referenced by a subpass:
    //           pInputAttachments: Attachments that are read from a shader;
    //         pResolveAttachments: Attachments used for multisampling color attachments;
    //     pDepthStencilAttachment: Attachment for depth and stencil data;
    //        pPreserveAttachments: Attachments that are not used by this subpass,
    //                              but for which the data must be preserved.
    VkSubpassDescription subpass
            {
                    0U,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    0U,
                    nullptr,
                    1U,
                    &colorAttachmentRef,
                    nullptr,
                    nullptr,
                    0U,
                    nullptr
            };

    VkSubpassDependency dependency
            {
                    VK_SUBPASS_EXTERNAL,
                    0U,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0U,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    0U
            };

    // The index of the color attachment (0-th) in array pAttachments
    // is directly referenced from the fragment shader
    // with the `layout(location = 0) out vec4 outColor` directive!
    VkRenderPassCreateInfo renderPassInfo
            {
                    VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                    nullptr,
                    0U,
                    1U,
                    &colorAttachment,
                    1U,
                    &subpass,
                    1U,
                    &dependency
            };

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create render pass");
    }
}


VkShaderModule VulkanApp::createShaderModule(const std::vector<char> & code)
{
    VkShaderModuleCreateInfo createInfo
            {
                    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    nullptr,
                    0U,
                    code.size(),
                    reinterpret_cast<const uint32_t *>(code.data())
            };

    VkShaderModule shaderModule;

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module");
    }

    return shaderModule;
}


void VulkanApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (std::size_t i = 0; i != swapChainImageViews.size(); ++i)
    {
        VkImageView attachments[] {swapChainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo
        {
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            nullptr,
            0U,
            renderPass,
            1U,
            attachments,
            swapChainExtent.width,
            swapChainExtent.height,
            1U
        };

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i])
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create framebuffer");
        }
    }
}


void VulkanApp::createCommandPool()
{
    QueueFamilyIndex queueFamilyIndex = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo
            {
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    nullptr,
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queueFamilyIndex.graphicsFamily.value()
            };

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command pool");
    }
}


void VulkanApp::createCommandBuffers()
{
    commandBuffers.resize(kMaxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo
            {
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    nullptr,
                    commandPool,
                    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    static_cast<uint32_t>(commandBuffers.size())
            };

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate command buffers");
    }
}


void VulkanApp::recordCommandBuffer(VkCommandBuffer commandBuffer_, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo commandBufferBeginInfo
            {
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    nullptr,
                    0U,
                    nullptr
            };

    if (vkBeginCommandBuffer(commandBuffer_, &commandBufferBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording command buffer");
    }

    VkClearValue clearValue {{{0.0f, 0.0f, 0.0f, 1.0f}}};

    VkRenderPassBeginInfo renderPassBeginInfo
    {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        nullptr,
        renderPass,
        swapChainFramebuffers[imageIndex],
        {{0, 0}, swapChainExtent},
        1U,
        &clearValue
    };

    vkCmdBeginRenderPass(commandBuffer_, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport
            {
                    0.0f,
                    0.0f,
                    static_cast<float>(swapChainExtent.width),
                    static_cast<float>(swapChainExtent.height),
                    0.0f,
                    1.0f
            };

    vkCmdSetViewport(commandBuffer_, 0U, 1U, &viewport);

    VkRect2D scissor
            {
                    {0, 0},
                    swapChainExtent
            };

    vkCmdSetScissor(commandBuffer_, 0U, 1U, &scissor);

    // commandBuffer: Self-explanatory.
    //   vertexCount: Even though we don't have a vertex buffer, we technically still have 3 vertices to drawFrame.
    // instanceCount: Used for instanced rendering, use 1 if you're not doing that.
    //   firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.
    // firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
    vkCmdDraw(commandBuffer_, 3U, 1U, 0U, 0U);

    vkCmdEndRenderPass(commandBuffer_);

    if (vkEndCommandBuffer(commandBuffer_) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to record command buffer");
    }
}


void VulkanApp::createSyncObjects()
{
    imageAvailableSemaphores.resize(kMaxFramesInFlight);
    renderFinishedSemaphores.resize(kMaxFramesInFlight);
    inFlightFences.resize(kMaxFramesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo
            {
                    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                    nullptr,
                    0U
            };

    VkFenceCreateInfo fenceInfo
            {
                    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    nullptr,
                    VK_FENCE_CREATE_SIGNALED_BIT
            };

    for (std::size_t i = 0; i != kMaxFramesInFlight; ++i)
    {
        if (vkCreateSemaphore(device,
                              &semaphoreInfo,
                              nullptr,
                              &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device,
                              &semaphoreInfo,
                              nullptr,
                              &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device,
                          &fenceInfo,
                          nullptr,
                          &inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create semaphores or fence");
        }
    }
}


void VulkanApp::glfwCleanup()
{
    glfwDestroyWindow(pWindow);
    glfwTerminate();
}


void VulkanApp::vulkanCleanup()
{
    cleanupSwapChain();

    for (std::size_t i = 0; i != kMaxFramesInFlight; ++i)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    // Remember, because command buffers are freed for us when we free the command pool,
    // there is nothing extra to do for command buffer cleanup.
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    vkDestroyDevice(device, nullptr);

    // Destory window surface and debug messenger before destorying Vulkan instance
    vkDestroySurfaceKHR(instance, surface, nullptr);

    if (enableValidationLayers)
    {
        destroyDebugUtilsMessengerEXT(nullptr);
    }

    vkDestroyInstance(instance, nullptr);
}


void VulkanApp::drawFrame(std::size_t frameIndex)
{
    static constexpr uint64_t kNeverTimeOut {std::numeric_limits<uint64_t>::max()};

    vkWaitForFences(device, 1, &inFlightFences[frameIndex], VK_TRUE, kNeverTimeOut);

    uint32_t imageIndex;

    VkResult result = vkAcquireNextImageKHR(device,
                                            swapChain,
                                            kNeverTimeOut,
                                            imageAvailableSemaphores[frameIndex],
                                            VK_NULL_HANDLE,
                                            &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    vkResetFences(device, 1, &inFlightFences[frameIndex]);

    vkResetCommandBuffer(commandBuffers[frameIndex], 0U);
    recordCommandBuffer(commandBuffers[frameIndex], imageIndex);

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[frameIndex]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[frameIndex]};

    VkSubmitInfo submitInfo
            {
                    VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    nullptr,
                    1U,
                    waitSemaphores,
                    waitStages,
                    1U,
                    &commandBuffers[frameIndex],
                    1U,
                    signalSemaphores
            };

    if (vkQueueSubmit(graphicsQueue,
                      1U,
                      &submitInfo,
                      inFlightFences[frameIndex]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit drawFrame command buffer");
    }

    VkSwapchainKHR swapChains[] = {swapChain};

    VkPresentInfoKHR presentInfo
            {
                    VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                    nullptr,
                    1U,
                    signalSemaphores,
                    1U,
                    swapChains,
                    &imageIndex,
                    nullptr
            };

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    // It is important to do this after vkQueuePresentKHR
    // to ensure that the semaphores are in a consistent state,
    // otherwise a signaled semaphore may never be properly waited upon.
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image");
    }
}


void VulkanApp::recreateSwapChain()
{
    // It is possible for the window surface to change such that
    // the swap chain is no longer compatible with it.
    // One of the reasons that could cause this to happen is the size of the window changing.
    // We have to catch these events and recreate the swap chain.

    // There is another case where a swap chain may become out of date
    // and that is a special kind of window resizing: window minimization.
    // This case is special because it will result in a frame buffer size of 0.
    int width = 0, height = 0;

    glfwGetFramebufferSize(pWindow, &width, &height);

    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(pWindow, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createFramebuffers();
}


void VulkanApp::cleanupSwapChain()
{
    for (auto swapChainFramebuffer : swapChainFramebuffers)
    {
        vkDestroyFramebuffer(device, swapChainFramebuffer, nullptr);
    }

    for (auto swapChainImageView : swapChainImageViews)
    {
        vkDestroyImageView(device, swapChainImageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
}


void VulkanApp::perFrameTimeLogic()
{
    double currentFrameTimeStamp = glfwGetTime();
    timeSinceLastFrame = currentFrameTimeStamp - lastFrameTimeStamp;
    lastFrameTimeStamp = currentFrameTimeStamp;
}


void VulkanApp::processKeyInput()
{
    // Camera control
    if (glfwGetKey(pWindow, GLFW_KEY_A) == GLFW_PRESS)
    {

    }

    if (glfwGetKey(pWindow, GLFW_KEY_D) == GLFW_PRESS)
    {

    }

    if (glfwGetKey(pWindow, GLFW_KEY_S) == GLFW_PRESS)
    {

    }

    if (glfwGetKey(pWindow, GLFW_KEY_W) == GLFW_PRESS)
    {

    }

    if (glfwGetKey(pWindow, GLFW_KEY_UP) == GLFW_PRESS)
    {

    }

    if (glfwGetKey(pWindow, GLFW_KEY_DOWN) == GLFW_PRESS)
    {

    }
}
