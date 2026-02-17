use crate::{
    Buffer, CommandQueue, ComputePipeline, Function, Library, MTLResourceOptions, MetalKernelError,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice};
use std::{ffi::c_void, ptr};

/// Metal device type classification based on Apple Silicon architecture.
///
/// MLX uses the last character of the architecture name to determine device type:
/// - 'p': phone (iPhone, small device)
/// - 'g': base/pro (M1/M2/M3 base and Pro variants)
/// - 's': max (M1/M2/M3 Max)
/// - 'd': ultra (M1/M2 Ultra)
///
/// Reference: refs/mlx/mlx/backend/metal/device.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalDeviceType {
    /// Small device (iPhone, 'p' suffix)
    Phone,
    /// Base/Pro device (M1/M2/M3 base and Pro, 'g' suffix)
    BasePro,
    /// Max device (M1/M2/M3 Max, 's' suffix)
    Max,
    /// Ultra device (M1/M2 Ultra, 'd' suffix)
    Ultra,
    /// Unknown or medium device (default)
    Medium,
}

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl AsRef<ProtocolObject<dyn MTLDevice>> for Device {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

impl Device {
    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    pub fn all() -> Vec<Self> {
        MTLCreateSystemDefaultDevice()
            .into_iter()
            .map(|raw| Device { raw })
            .collect()
    }

    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice().map(|raw| Device { raw })
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(Buffer::new)
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Buffer".to_string(),
            ))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void).unwrap();
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(Buffer::new)
                .ok_or(MetalKernelError::FailedToCreateResource(
                    "Buffer".to_string(),
                ))
        }
    }

    /// Compile a Metal library from source, retrying on transient XPC failures.
    ///
    /// On iOS the Metal shader compiler runs in an XPC service
    /// (`com.apple.MTLCompilerService`). Under memory pressure iOS can kill
    /// this service, producing `XPC_ERROR_CONNECTION_INTERRUPTED`. The service
    /// restarts automatically, so a retry after a short delay usually succeeds.
    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library, MetalKernelError> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(200);

        let mut last_err = None;
        for attempt in 0..MAX_RETRIES {
            match self
                .as_ref()
                .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            {
                Ok(raw) => return Ok(Library::new(raw)),
                Err(e) => {
                    let msg = e.to_string();
                    if attempt + 1 < MAX_RETRIES && msg.contains("XPC") {
                        tracing::warn!(
                            "Metal shader compilation failed (attempt {}/{}): {}, retrying...",
                            attempt + 1,
                            MAX_RETRIES,
                            msg
                        );
                        std::thread::sleep(RETRY_DELAY);
                    }
                    last_err = Some(msg);
                }
            }
        }
        Err(MetalKernelError::LoadLibraryError(last_err.unwrap_or_default()))
    }

    /// Load a pre-compiled Metal library (.metallib) from a file path.
    ///
    /// Pre-compiled libraries bypass runtime shader compilation entirely,
    /// avoiding XPC failures on iOS. Build metallib files with:
    /// ```sh
    /// xcrun -sdk iphoneos metal -c shader.metal -o shader.air
    /// xcrun -sdk iphoneos metallib shader.air -o shader.metallib
    /// ```
    pub fn new_library_with_url(
        &self,
        path: &std::path::Path,
    ) -> Result<Library, MetalKernelError> {
        let path_str = path.to_string_lossy();
        let url_str = format!("file://{}", path_str);
        let ns_url_str = NSString::from_str(&url_str);
        let url = NSURL::URLWithString(&ns_url_str)
            .ok_or_else(|| MetalKernelError::LoadLibraryError(
                format!("Invalid metallib path: {}", path_str),
            ))?;

        let raw = self
            .as_ref()
            .newLibraryWithURL_error(&url)
            .map_err(|e| MetalKernelError::LoadLibraryError(
                format!("Failed to load metallib '{}': {}", path_str, e),
            ))?;

        Ok(Library::new(raw))
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let raw = self
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
        Ok(ComputePipeline::new(raw))
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, MetalKernelError> {
        let raw = self
            .as_ref()
            .newCommandQueue()
            .ok_or(MetalKernelError::FailedToCreateResource("CommandQueue".to_string()))?;
        Ok(raw)
    }

    pub fn recommended_max_working_set_size(&self) -> usize {
        self.as_ref().recommendedMaxWorkingSetSize() as usize
    }

    pub fn current_allocated_size(&self) -> usize {
        self.as_ref().currentAllocatedSize()
    }

    /// Get the device architecture name (e.g., "applegpu_g13g", "applegpu_g14d").
    ///
    /// This returns the full architecture string from the Metal device.
    /// The last character indicates the device type:
    /// - 'p': phone
    /// - 'g': base/pro
    /// - 's': max
    /// - 'd': ultra
    pub fn architecture_name(&self) -> String {
        let arch = self.as_ref().architecture();
        arch.name().to_string()
    }

    /// Get the device type based on architecture name.
    ///
    /// This implements the same logic as MLX's device type detection.
    /// Reference: refs/mlx/mlx/backend/metal/device.cpp
    pub fn device_type(&self) -> MetalDeviceType {
        let arch = self.architecture_name();
        match arch.chars().last() {
            Some('p') => MetalDeviceType::Phone,
            Some('g') => MetalDeviceType::BasePro,
            Some('s') => MetalDeviceType::Max,
            Some('d') => MetalDeviceType::Ultra,
            _ => MetalDeviceType::Medium,
        }
    }
}
