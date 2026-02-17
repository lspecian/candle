pub const AFFINE: &str = include_str!("metal_src/affine.metal");
pub const BINARY: &str = include_str!("metal_src/binary.metal");
pub const CAST: &str = include_str!("metal_src/cast.metal");
pub const CONV: &str = include_str!("metal_src/conv.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
pub const INDEXING: &str = include_str!("metal_src/indexing.metal");
pub const MLX_GEMM: &str = include_str!("metal_src/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");
pub const RANDOM: &str = include_str!("metal_src/random.metal");
pub const REDUCE: &str = include_str!("metal_src/reduce.metal");
pub const SORT: &str = include_str!("metal_src/sort.metal");
pub const TERNARY: &str = include_str!("metal_src/ternary.metal");
pub const UNARY: &str = include_str!("metal_src/unary.metal");
pub const SDPA: &str = include_str!("metal_src/scaled_dot_product_attention.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Gemm,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Sdpa,
}

impl Source {
    /// Returns the metallib filename for this source (e.g. "quantized.metallib").
    pub fn metallib_filename(&self) -> &'static str {
        match self {
            Source::Affine => "affine.metallib",
            Source::Binary => "binary.metallib",
            Source::Cast => "cast.metallib",
            Source::Conv => "conv.metallib",
            Source::Fill => "fill.metallib",
            Source::Gemm => "mlx_gemm.metallib",
            Source::Indexing => "indexing.metallib",
            Source::MlxSort => "mlx_sort.metallib",
            Source::Quantized => "quantized.metallib",
            Source::Random => "random.metallib",
            Source::Reduce => "reduce.metallib",
            Source::Sort => "sort.metallib",
            Source::Ternary => "ternary.metallib",
            Source::Unary => "unary.metallib",
            Source::Sdpa => "scaled_dot_product_attention.metallib",
        }
    }
}
