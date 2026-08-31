"""验证 CableDefectDataset 能正确加载并产出张量"""
import sys
sys.path.insert(0, r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\Src")

from core.dataset import CableDefectDataset

ds = CableDefectDataset(
    manifest_path=r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\DataSet\manifest.yaml",
    split="train",
    channels=["impulse", "step"],
)

print(f"数据集: {len(ds)} 样本")

# 测试前3个样本
for i in range(min(3, len(ds))):
    inputs, labels = ds[i]
    print(f"\n样本 {i}:")
    print(f"  inputs shape: {inputs.shape}, dtype={inputs.dtype}")
    print(f"  labels shape: {labels.shape}, dtype={labels.dtype}")
    print(f"  inputs range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"  labels max: {labels.max():.4f}, 非零比例: {(labels > 0.01).float().mean():.4f}")

# 测试 DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True)
batch_inputs, batch_labels = next(iter(loader))
print(f"\nDataLoader batch:")
print(f"  batch_inputs shape: {batch_inputs.shape}")
print(f"  batch_labels shape: {batch_labels.shape}")

print("\n✅ PyTorch Dataset 验证通过")
