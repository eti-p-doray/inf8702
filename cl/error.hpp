#pragma once

#include <exception>
#include <sstream>

#include "cl/cl.hpp"

namespace cl {

class opencl_error : public std::exception {
 public:
  /// Creates a new opencl_error exception object for \p error.
  explicit opencl_error(cl_int error) noexcept
      : m_error(error),
        m_error_string(to_string(error)) {}

  /// Destroys the opencl_error object.
  ~opencl_error() {}

  /// Returns the numeric error code.
  cl_int error_code() const noexcept {
    return m_error;
  }

  /// Returns a string description of the error.
  std::string error_string() const noexcept {
    return m_error_string;
  }

  /// Returns a C-string description of the error.
  const char* what() const noexcept {
    return m_error_string.c_str();
  }

  /// Static function which converts the numeric OpenCL error code \p error
  /// to a human-readable string.
  ///
  /// For example:
  /// \code
  /// std::cout << opencl_error::to_string(CL_INVALID_KERNEL_ARGS) << std::endl;
  /// \endcode
  ///
  /// Will print "Invalid Kernel Arguments".
  ///
  /// If the error code is unknown (e.g. not a valid OpenCL error), a string
  /// containing "Unknown OpenCL Error" along with the error number will be
  /// returned.
  static std::string to_string(cl_int error) {
    switch(error) {
      case CL_SUCCESS: return "Success";
      case CL_DEVICE_NOT_FOUND: return "Device Not Found";
      case CL_DEVICE_NOT_AVAILABLE: return "Device Not Available";
      case CL_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
      case CL_OUT_OF_RESOURCES: return "Out of Resources";
      case CL_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
      case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
      case CL_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
      case CL_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
      case CL_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
      case CL_MAP_FAILURE: return "Map Failure";
      case CL_INVALID_VALUE: return "Invalid Value";
      case CL_INVALID_DEVICE_TYPE: return "Invalid Device Type";
      case CL_INVALID_PLATFORM: return "Invalid Platform";
      case CL_INVALID_DEVICE: return "Invalid Device";
      case CL_INVALID_CONTEXT: return "Invalid Context";
      case CL_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
      case CL_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
      case CL_INVALID_HOST_PTR: return "Invalid Host Pointer";
      case CL_INVALID_MEM_OBJECT: return "Invalid Memory Object";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
      case CL_INVALID_IMAGE_SIZE: return "Invalid Image Size";
      case CL_INVALID_SAMPLER: return "Invalid Sampler";
      case CL_INVALID_BINARY: return "Invalid Binary";
      case CL_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
      case CL_INVALID_PROGRAM: return "Invalid Program";
      case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
      case CL_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
      case CL_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
      case CL_INVALID_KERNEL: return "Invalid Kernel";
      case CL_INVALID_ARG_INDEX: return "Invalid Argument Index";
      case CL_INVALID_ARG_VALUE: return "Invalid Argument Value";
      case CL_INVALID_ARG_SIZE: return "Invalid Argument Size";
      case CL_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
      case CL_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
      case CL_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
      case CL_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
      case CL_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
      case CL_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
      case CL_INVALID_EVENT: return "Invalid Event";
      case CL_INVALID_OPERATION: return "Invalid Operation";
      case CL_INVALID_GL_OBJECT: return "Invalid GL Object";
      case CL_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
      case CL_INVALID_MIP_LEVEL: return "Invalid MIP Level";
      case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
      case CL_COMPILE_PROGRAM_FAILURE: return "Compile Program Failure";
      case CL_LINKER_NOT_AVAILABLE: return "Linker Not Available";
      case CL_LINK_PROGRAM_FAILURE: return "Link Program Failure";
      case CL_DEVICE_PARTITION_FAILED: return "Device Partition Failed";
      case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "Kernel Argument Info Not Available";
      case CL_INVALID_PROPERTY: return "Invalid Property";
      case CL_INVALID_IMAGE_DESCRIPTOR: return "Invalid Image Descriptor";
      case CL_INVALID_COMPILER_OPTIONS: return "Invalid Compiler Options";
      case CL_INVALID_LINKER_OPTIONS: return "Invalid Linker Options";
      case CL_INVALID_DEVICE_PARTITION_COUNT: return "Invalid Device Partition Count";
      default: {
          std::stringstream s;
          s << "Unknown OpenCL Error (" << error << ")";
          return s.str();
      }
    }
  }

private:
  cl_int m_error;
  std::string m_error_string;
};

inline cl_int check_error(cl_int error) {
  if (error != CL_SUCCESS)
    throw opencl_error(error);
  return error;
}

}
