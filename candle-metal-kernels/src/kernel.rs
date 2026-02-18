/// Safe stderr logging — never panics if stderr is unavailable (iOS).
macro_rules! safe_log {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

use crate::source::{
    AFFINE, BINARY, CAST, CONV, FILL, INDEXING, MLX_GEMM, MLX_SORT, QUANTIZED, RANDOM, REDUCE,
    SDPA, SORT, TERNARY, UNARY,
};
use crate::utils::get_env_bool;
use crate::{
    ComputePipeline, ConstantValues, Device, Function, Library, MTLCompileOptions,
    MTLMathFloatingPointFunctions, MTLMathMode, MetalKernelError, Source,
};
use objc2::available;
use objc2::rc::Retained;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub enum KernelName {
    Ref(&'static str),
    Value(String),
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::Ref(r) => r,
            Self::Value(v) => v.as_str(),
        }
    }
}

impl std::hash::Hash for KernelName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Ref(r) => r.hash(state),
            Self::Value(v) => v.hash(state),
        }
    }
}

impl PartialEq for KernelName {
    fn eq(&self, other: &Self) -> bool {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1 == v2
    }
}

impl Eq for KernelName {}

impl From<&'static str> for KernelName {
    fn from(value: &'static str) -> Self {
        Self::Ref(value)
    }
}

impl From<String> for KernelName {
    fn from(value: String) -> Self {
        Self::Value(value)
    }
}

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(KernelName, Option<ConstantValues>), ComputePipeline>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
    /// Optional directory containing pre-compiled .metallib files.
    /// When set, `load_library` tries loading from here before falling back
    /// to runtime source compilation.
    metallib_dir: RwLock<Option<PathBuf>>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
            metallib_dir: RwLock::new(None),
        }
    }

    /// Set the directory containing pre-compiled `.metallib` files.
    ///
    /// When set, `load_library` will attempt to load the pre-compiled binary
    /// before falling back to runtime source compilation. This eliminates
    /// XPC-based shader compilation, which is critical for iOS reliability.
    ///
    /// Build metallib files from the Metal sources with:
    /// ```sh
    /// for f in *.metal; do
    ///   xcrun -sdk iphoneos metal -c "$f" -o "${f%.metal}.air"
    ///   xcrun -sdk iphoneos metallib "${f%.metal}.air" -o "${f%.metal}.metallib"
    /// done
    /// ```
    pub fn set_metallib_dir(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        safe_log!("[Candle] set_metallib_dir: {}", path.display());
        *self.metallib_dir.write().unwrap() = Some(path);
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::Fill => FILL,
            Source::Gemm => MLX_GEMM,
            Source::Indexing => INDEXING,
            Source::MlxSort => MLX_SORT,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
            Source::Sort => SORT,
            Source::Ternary => TERNARY,
            Source::Unary => UNARY,
            Source::Sdpa => SDPA,
        }
    }

    /// Load the given library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    ///
    /// When a `metallib_dir` is configured, tries loading the pre-compiled
    /// `.metallib` first. Falls back to runtime source compilation if the
    /// metallib is not found or fails to load.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            // Try pre-compiled metallib first
            let metallib_result = self.try_load_metallib(device, source);
            let lib = match metallib_result {
                Some(Ok(lib)) => lib,
                Some(Err(e)) => {
                    tracing::warn!(
                        "Failed to load pre-compiled metallib for {:?}: {}, falling back to source compilation",
                        source, e
                    );
                    self.compile_from_source(device, source)?
                }
                None => self.compile_from_source(device, source)?,
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    /// Try loading a pre-compiled .metallib file if a metallib_dir is set.
    fn try_load_metallib(
        &self,
        device: &Device,
        source: Source,
    ) -> Option<Result<Library, MetalKernelError>> {
        let dir_guard = self.metallib_dir.read().ok()?;
        let dir = dir_guard.as_ref()?;
        let filename = source.metallib_filename();
        let path = dir.join(filename);
        if path.exists() {
            safe_log!("[Candle] Loading pre-compiled metallib: {}", path.display());
            let result = device.new_library_with_url(&path);
            if let Err(ref e) = result {
                safe_log!("[Candle] FAILED to load metallib {}: {}", filename, e);
            } else {
                safe_log!("[Candle] OK loaded metallib: {}", filename);
            }
            Some(result)
        } else {
            safe_log!("[Candle] metallib not found: {} (dir={})", path.display(), dir.display());
            None
        }
    }

    /// Compile a library from embedded Metal source.
    fn compile_from_source(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        safe_log!("[Candle] Compiling {:?} from source (runtime XPC compilation)...", source);
        let source_content = self.get_library_source(source);
        let compile_options = get_compile_options();
        let result = device
            .new_library_with_source(source_content, Some(&compile_options))
            .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()));
        match &result {
            Ok(_) => safe_log!("[Candle] Compiled {:?} OK", source),
            Err(e) => safe_log!("[Candle] FAILED to compile {:?}: {}", source, e),
        }
        result
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &str,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    pub fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name.into(), constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(device, source, name.as_ref(), constants.as_ref())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}

fn get_compile_options() -> Retained<MTLCompileOptions> {
    let compile_options = MTLCompileOptions::new();
    //unsafe { compile_options.setEnableLogging(true) };

    let fast_math_enabled = get_env_bool("CANDLE_METAL_ENABLE_FAST_MATH", true);
    // Ref availability:
    // https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode
    if available!(macos = 15, ios = 18) {
        if fast_math_enabled {
            compile_options.setMathMode(MTLMathMode::Fast);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        } else {
            compile_options.setMathMode(MTLMathMode::Relaxed);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
    } else {
        // For older OS versions we use the old api
        #[allow(deprecated)]
        compile_options.setFastMathEnabled(fast_math_enabled);
    }
    compile_options
}
