#ifndef SHADER_H
#define SHADER_H

#include <optional>
#include <stdexcept>

#include <shaderc/shaderc.hpp>


class Shader
{
public:
    Shader()
    {
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
    }

private:
    // Returns GLSL shader source text after preprocessing.
    std::optional<std::string>
    preprocess_shader(const std::string & name, shaderc_shader_kind kind, const std::string & text)
    {
        shaderc::PreprocessedSourceCompilationResult result =
                compiler.PreprocessGlsl(text,
                                        kind,
                                        name.c_str(),
                                        options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(result.GetErrorMessage());
            return {};
        }

        return {{result.cbegin(), result.cend()}};
    }

    // Compiles a shader to SPIR-V assembly.
    // Returns the assembly text as a string.
    std::optional<std::string>
    compile_file_to_assembly(const std::string & name, shaderc_shader_kind kind, const std::string & text)
    {
        shaderc::AssemblyCompilationResult result =
                compiler.CompileGlslToSpvAssembly(text,
                                                  kind,
                                                  name.c_str(),
                                                  options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(result.GetErrorMessage());
            return {};
        }

        return {{result.cbegin(), result.cend()}};
    }

    // Compiles a shader to a SPIR-V binary.
    // Returns the binary as a vector of 32-bit words.
    std::optional<std::vector<uint32_t>>
    compile_file(const std::string & name, shaderc_shader_kind kind, const std::string & text)
    {
        shaderc::SpvCompilationResult module =
                compiler.CompileGlslToSpv(text,
                                          kind,
                                          name.c_str(),
                                          options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(module.GetErrorMessage());
            return {};
        }

        return {{module.cbegin(), module.cend()}};
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
};


#endif  // SHADER_H
