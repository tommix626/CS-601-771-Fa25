# Self-Attention Profiling Experiment

This project implements a comprehensive profiling system for Self-Attention mechanisms in PyTorch, measuring computational complexity, memory usage, and wall clock time across different input sequence lengths.

## Features

- **Complete Self-Attention Implementation**: Multi-head attention mechanism with proper linear transformations
- **Comprehensive Profiling**: Measures FLOPS, memory usage, and wall clock time
- **Error Analysis**: Includes standard error bars for statistical significance
- **Multi-Device Support**: Runs on both CPU and GPU (if available)
- **Visualization**: Creates detailed plots with error bars using seaborn
- **Theoretical Analysis**: Compares empirical results with theoretical FLOPS calculations

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the profiling experiment:

```bash
python profiling_self_attention.py
```

## Configuration

The experiment is configured with the following parameters:

- **Sequence Lengths**: [10, 100, 1000, 10000]
- **Model Dimension**: 512
- **Number of Heads**: 8
- **Number of Runs**: 20 (for averaging and error calculation)

## Output

The script generates:

1. **Console Output**: Detailed analysis with mean values and standard errors
2. **Visualization**: `self_attention_profiling.png` with 4 subplots:
   - Wall Clock Time vs Sequence Length
   - Memory Usage vs Sequence Length  
   - Theoretical FLOPS vs Sequence Length
   - Time vs Computational Complexity
3. **Data File**: `profiling_results.npz` containing all raw data

## Key Measurements

### 1. Computational Complexity (FLOPS)
- **Linear Transformations**: 3 × seq_len × d_model × d_model
- **Q×K^T**: seq_len × seq_len × d_k × num_heads
- **Attention×V**: seq_len × seq_len × d_k × num_heads
- **Output Projection**: seq_len × d_model × d_model

### 2. Memory Usage
- GPU: CUDA memory allocation tracking
- CPU: Process memory usage via psutil

### 3. Wall Clock Time
- Precise timing with proper synchronization
- Multiple runs for statistical significance

## Expected Results

The profiling will demonstrate:

- **O(n²) Complexity**: Self-attention scales quadratically with sequence length
- **Memory Growth**: Memory usage increases with sequence length
- **GPU vs CPU**: Performance comparison between devices
- **Error Bars**: Statistical significance of measurements

## Code Structure

- `SelfAttention`: PyTorch module implementing multi-head attention
- `profile_self_attention()`: Core profiling function
- `count_flops_self_attention()`: Theoretical FLOPS calculation
- `plot_results()`: Visualization with error bars
- `run_profiling_experiment()`: Main experiment orchestration

## GitHub Repository

The complete code is available at: [GitHub Repository Link]

## Interpretation

The results will show that Self-Attention has:
- **Quadratic time complexity** O(n²) due to the attention matrix computation
- **Quadratic memory complexity** O(n²) for storing attention weights
- **Significant performance difference** between CPU and GPU implementations
- **Statistical reliability** through proper error analysis

This profiling provides empirical evidence of the computational challenges of scaling Self-Attention to very long sequences, which has led to the development of various attention variants (e.g., Linear Attention, Sparse Attention) in recent research. 