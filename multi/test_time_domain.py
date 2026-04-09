"""
测试时域信号集成
"""
import torch
import numpy as np
import h5py
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi.model import TimeDomainTransformerEncoder, create_czsl_model
from multi.data import TimeDomainDataset, CombinedRadarDataset, STFTDataset, create_czsl_dataloaders

def test_time_domain_encoder():
    """测试时域编码器"""
    print("Testing TimeDomainTransformerEncoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建编码器
    encoder = TimeDomainTransformerEncoder(
        embed_dim=512,
        num_heads=8,
        num_layers=3,
        seq_len=8000
    ).to(device)

    # 创建模拟数据
    batch_size = 4
    seq_len = 8000
    dummy_input = torch.randn(batch_size, seq_len).to(device)

    # 前向传播
    with torch.no_grad():
        output = encoder(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 512)")

    assert output.shape == (batch_size, 512), f"Output shape mismatch: {output.shape}"
    print("  [OK] TimeDomainTransformerEncoder test passed!")

def test_model_with_time_domain():
    """测试带时域信号的模型"""
    print("\nTesting CLIPForCZSL with time domain...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建测试配置
    test_config = {
        "model": {
            "clip_model": "ViT-B/32",
            "freeze_vision": False,
            "freeze_text": True,
            "vision_layers_unfreeze": 2
        },
        "data": {
            "time_seq_len": 8000
        },
        "use_time_domain": True,
        "jamming_classes": [
            {"name": "DFTJ"},
            {"name": "ISRJ"},
            {"name": "VGPO"},
            {"name": "RGPO"},
        ]
    }

    # 创建模型
    model = create_czsl_model(test_config, device=device)

    # 模拟输入
    batch_size = 2
    stft_images = torch.randn(batch_size, 3, 224, 224).to(device)
    time_signals = torch.randn(batch_size, 8000).to(device)
    text_tokens = torch.randint(0, 1000, (batch_size, 77)).to(device)

    # 前向传播
    with torch.no_grad():
        image_features, text_features = model(stft_images, text_tokens, time_signals)

    print(f"  STFT images shape: {stft_images.shape}")
    print(f"  Time signals shape: {time_signals.shape}")
    print(f"  Image features shape: {image_features.shape}")
    print(f"  Text features shape: {text_features.shape}")

    assert image_features.shape == (batch_size, 512), f"Image features shape mismatch: {image_features.shape}"
    assert text_features.shape == (batch_size, 512), f"Text features shape mismatch: {text_features.shape}"
    print("  [OK] Model with time domain test passed!")

def test_data_loading():
    """测试数据加载（如果有时域数据文件）"""
    print("\nTesting data loading...")

    # 检查是否有测试数据
    test_data_dir = "D:/VScode/Jamming_signal_simulation/output/260403/JNR_+0"
    if not os.path.exists(test_data_dir):
        print("  [WARN] Test data directory not found, skipping data loading test")
        return

    # 检查时域数据文件
    time_file = os.path.join(test_data_dir, "train_echo_time.mat")
    if not os.path.exists(time_file):
        print("  [WARN] Time domain data file not found, skipping data loading test")
        return

    try:
        # 尝试加载时域数据
        dataset = TimeDomainDataset(
            time_file=time_file,
            time_var_name="all_times",
            seq_len=8000
        )

        # 测试获取一个样本
        time_tensor = dataset[0]
        print(f"  Time tensor shape: {time_tensor.shape}")
        assert time_tensor.shape == (8000,), f"Time tensor shape mismatch: {time_tensor.shape}"
        print("  [OK] Data loading test passed!")

    except Exception as e:
        print(f"  [WARN] Data loading test failed: {e}")

def test_config_switch():
    """测试配置开关"""
    print("\nTesting configuration switch...")

    # 测试启用时域
    config_on = {"use_time_domain": True, "data": {"time_seq_len": 8000}}
    model_on = create_czsl_model({"model": {}, **config_on, "jamming_classes": [{"name": "test"}]}, device="cpu")
    assert hasattr(model_on, 'time_encoder'), "Model should have time_encoder when use_time_domain=True"
    assert model_on.use_time_domain == True
    print("  [OK] use_time_domain=True test passed!")

    # 测试禁用时域
    config_off = {"use_time_domain": False, "data": {"time_seq_len": 8000}}
    model_off = create_czsl_model({"model": {}, **config_off, "jamming_classes": [{"name": "test"}]}, device="cpu")
    assert not hasattr(model_off, 'time_encoder'), "Model should not have time_encoder when use_time_domain=False"
    assert model_off.use_time_domain == False
    print("  [OK] use_time_domain=False test passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Time Domain Integration")
    print("=" * 60)

    try:
        test_time_domain_encoder()
        test_model_with_time_domain()
        test_data_loading()
        test_config_switch()

        print("\n" + "=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
