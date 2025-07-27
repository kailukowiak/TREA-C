"""Test memory usage with large batch sizes and patch sizes."""

import torch
import gc
from duet.models import PatchTSTNan


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        # Rough estimate for CPU
        import psutil
        return psutil.Process().memory_info().rss / 1024**2


def test_memory_config(batch_size, patch_size, C_num=20, T=64, d_model=128):
    """Test memory usage for a specific configuration."""
    print(f"\n--- Testing: batch_size={batch_size}, patch_size={patch_size} ---")
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_memory = get_memory_usage()
    
    try:
        # Create model
        model = PatchTSTNan(
            C_num=C_num,
            C_cat=0,
            T=T,
            d_model=d_model,
            patch_size=patch_size,
            num_classes=3,
            n_heads=8,
            n_layers=4,
            task='classification'
        )
        
        model_memory = get_memory_usage()
        
        # Create sample batch
        x_num = torch.randn(batch_size, C_num, T)
        x_cat = torch.empty(batch_size, 0, T, dtype=torch.long)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()
        
        data_memory = get_memory_usage()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x_num, x_cat)
        
        forward_memory = get_memory_usage()
        
        # Calculate memory usage
        model_size = model_memory - start_memory
        data_size = data_memory - model_memory
        forward_size = forward_memory - data_memory
        total_size = forward_memory - start_memory
        
        print(f"✅ SUCCESS")
        print(f"  Model memory: {model_size:.1f} MB")
        print(f"  Data memory: {data_size:.1f} MB")
        print(f"  Forward pass memory: {forward_size:.1f} MB")
        print(f"  Total memory: {total_size:.1f} MB")
        print(f"  Output shape: {output.shape}")
        
        # Memory per sample
        memory_per_sample = total_size / batch_size
        print(f"  Memory per sample: {memory_per_sample:.2f} MB")
        
        return {
            'batch_size': batch_size,
            'patch_size': patch_size,
            'total_memory': total_size,
            'memory_per_sample': memory_per_sample,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {
            'batch_size': batch_size,
            'patch_size': patch_size,
            'total_memory': float('inf'),
            'memory_per_sample': float('inf'),
            'success': False
        }
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'x_num' in locals():
            del x_num
        if 'x_cat' in locals():
            del x_cat
        if 'output' in locals():
            del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("Memory Usage Testing for PatchTSTNan")
    print("=" * 50)
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Test configurations
    configs = [
        # (batch_size, patch_size)
        (128, 8),    # Original small
        (128, 32),   # Optimized medium
        (256, 32),   # Larger batch
        (512, 32),   # Even larger batch
        (1024, 32),  # Maximum batch (standard)
        (1024, 64),  # Ultra-fast configuration
    ]
    
    results = []
    
    for batch_size, patch_size in configs:
        result = test_memory_config(
            batch_size=batch_size,
            patch_size=patch_size,
            C_num=20,  # Estimated W3 features
            T=64 if patch_size <= 32 else 128,  # Sequence length
            d_model=128
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("MEMORY USAGE SUMMARY")
    print("=" * 60)
    print(f"{'Batch':<8} {'Patch':<8} {'Total MB':<12} {'MB/Sample':<12} {'Status':<10}")
    print("-" * 60)
    
    for result in results:
        status = "✅ OK" if result['success'] else "❌ FAIL"
        memory_str = f"{result['total_memory']:.1f}" if result['success'] else "N/A"
        per_sample_str = f"{result['memory_per_sample']:.2f}" if result['success'] else "N/A"
        
        print(f"{result['batch_size']:<8} {result['patch_size']:<8} "
              f"{memory_str:<12} {per_sample_str:<12} {status:<10}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    successful_configs = [r for r in results if r['success']]
    if successful_configs:
        best_config = max(successful_configs, key=lambda x: x['batch_size'])
        print(f"✅ Recommended configuration:")
        print(f"   batch_size={best_config['batch_size']}")
        print(f"   patch_size={best_config['patch_size']}")
        print(f"   Total memory: {best_config['total_memory']:.1f} MB")
        
        print(f"\n💡 Tips:")
        print(f"   - Start with batch_size={best_config['batch_size']}")
        print(f"   - If you get OOM errors, reduce batch size by half")
        print(f"   - Larger patch sizes = less memory per sample")
        print(f"   - Use mixed precision (16-bit) for additional memory savings")
    else:
        print("❌ No configurations fit in memory. Try:")
        print("   - Smaller batch sizes (64, 32)")
        print("   - Larger patch sizes (64, 128)")
        print("   - Smaller model dimensions")


if __name__ == "__main__":
    main()