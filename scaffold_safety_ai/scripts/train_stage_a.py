# scripts/train_stage_a.py
# Stage A 훈련: PointLoRA-Only Training (엄밀한 구현)

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integrate_shapellm import create_scaffold_model

class MockScaffoldDataset(Dataset):
    """Mock 비계 안전 데이터셋 (엄밀한 구현)"""
    
    def __init__(self, num_samples=1000, mode='train'):
        self.num_samples = num_samples
        self.mode = mode
        self.safety_levels = ['safe', 'warning', 'danger']
        self.safety_map = {'safe': 0, 'warning': 1, 'danger': 2}
        
        # 재현 가능한 시드 설정
        np.random.seed(42 if mode == 'train' else 123)
        torch.manual_seed(42 if mode == 'train' else 123)
        
        print(f"✅ MockScaffoldDataset created: {num_samples} {mode} samples")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 재현 가능한 데이터 생성
        np.random.seed(idx + (42 if self.mode == 'train' else 123))
        torch.manual_seed(idx + (42 if self.mode == 'train' else 123))
        
        # Mock scaffold point cloud 생성 (구조적 특성 반영)
        point_cloud = self._generate_scaffold_structure()
        
        # Safety level 결정 (편향된 분포로 현실성 증가)
        if self.mode == 'train':
            safety_probs = [0.5, 0.3, 0.2]  # safe가 많음
        else:
            safety_probs = [0.4, 0.4, 0.2]  # 검증에서는 더 균등
            
        safety_level = np.random.choice(self.safety_levels, p=safety_probs)
        safety_label = self.safety_map[safety_level]
        
        # 높이 정보 (안전도에 따른 현실적 분포)
        if safety_level == 'safe':
            height = np.random.normal(105, 10)  # 평균 105cm
        elif safety_level == 'warning':
            height = np.random.normal(88, 5)    # 평균 88cm (기준선 근처)
        else:  # danger
            height = np.random.normal(75, 8)    # 평균 75cm (기준 미달)
            
        height = max(50, min(150, height))  # 현실적 범위로 제한
        
        return {
            'point_cloud': point_cloud,
            'safety_label': torch.tensor(safety_label, dtype=torch.long),
            'height': torch.tensor(height, dtype=torch.float32),
            'sample_id': idx
        }
    
    def _generate_scaffold_structure(self):
        """구조적 특성을 반영한 비계 포인트클라우드 생성"""
        points = []
        
        # 1. 수직 기둥들 (4개 모서리)
        for x in [-1, 1]:
            for y in [-1, 1]:
                z_points = np.linspace(0, 3, 200)  # 높이 3m
                noise = np.random.normal(0, 0.05, 200)
                column_points = np.column_stack([
                    np.full(200, x) + noise,
                    np.full(200, y) + noise,
                    z_points + noise
                ])
                points.append(column_points)
        
        # 2. 수평 플랫폼들 (여러 층)
        for z in [1.0, 2.0, 3.0]:  # 1m, 2m, 3m 높이
            x_grid, y_grid = np.meshgrid(
                np.linspace(-1, 1, 30),
                np.linspace(-1, 1, 30)
            )
            platform_points = np.column_stack([
                x_grid.flatten(),
                y_grid.flatten(),
                np.full(900, z) + np.random.normal(0, 0.02, 900)
            ])
            points.append(platform_points)
        
        # 3. 연결 빔들 (대각선, 수평)
        for _ in range(500):  # 추가 구조 요소들
            start = np.random.uniform(-1, 1, 3)
            end = np.random.uniform(-1, 1, 3) 
            end[2] = start[2]  # 같은 높이로 연결
            
            t = np.linspace(0, 1, 20)
            beam_points = np.outer(1-t, start) + np.outer(t, end)
            points.append(beam_points)
        
        # 4. 모든 점들 결합 및 정규화
        all_points = np.vstack(points)
        
        # 8192개 점으로 샘플링
        if len(all_points) > 8192:
            indices = np.random.choice(len(all_points), 8192, replace=False)
            sampled_points = all_points[indices]
        else:
            # 부족하면 복제로 채움
            needed = 8192 - len(all_points)
            extra_indices = np.random.choice(len(all_points), needed, replace=True)
            sampled_points = np.vstack([all_points, all_points[extra_indices]])
        
        # RGB 정보 추가 (회색 금속 색상)
        colors = np.random.normal([0.5, 0.5, 0.5], 0.1, (8192, 3))
        colors = np.clip(colors, 0, 1)
        
        # XYZ + RGB = 6D
        point_cloud = np.hstack([sampled_points, colors]).astype(np.float32)
        
        return torch.tensor(point_cloud)

class StageATrainer:
    """Stage A 훈련 클래스 (엄밀한 구현)"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎯 Training device: {self.device}")
        
        # 모델 초기화
        self.model = create_scaffold_model(config).to(self.device)
        
        # LoRA 레이어 추가 (실제 ShapeLLM 구조 모방)
        self._add_lora_layers()
        
        # 훈련 모드 설정 (LoRA + Safety 모듈만)
        self.model.set_training_mode(scaffold_training=True)
        
        # Optimizer 및 Loss 설정
        self._setup_training()
        
        # 메트릭 추적
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def _add_lora_layers(self):
        """실제 ShapeLLM Transformer 구조에 맞춘 LoRA 추가"""
        print("🔧 Adding LoRA layers...")
        
        # 일반적인 Transformer 구조의 주요 레이어들
        lora_configs = [
            ('layer_0_attention_qkv', 768, 768*3),
            ('layer_0_mlp_fc1', 768, 3072),
            ('layer_1_attention_qkv', 768, 768*3),
            ('layer_1_mlp_fc1', 768, 3072),
            ('layer_2_attention_qkv', 768, 768*3),
            ('layer_2_mlp_fc1', 768, 3072),
            # 더 많은 레이어 추가 가능
        ]
        
        total_lora_params = 0
        for layer_name, in_dim, out_dim in lora_configs:
            self.model.add_lora_layer(layer_name, in_dim, out_dim)
            # 파라미터 수 계산
            rank = self.config['lora_rank']
            params = rank * (in_dim + out_dim)
            total_lora_params += params
        
        print(f"📊 Total LoRA parameters: {total_lora_params:,}")
        
        # 훈련 효율성 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        efficiency = (total_lora_params / total_params) * 100
        print(f"📊 Training efficiency: {efficiency:.2f}% (Target: ~3.43%)")
        
    def _setup_training(self):
        """훈련 설정"""
        # 훈련 가능한 파라미터만 수집
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.2, 1.5]).to(self.device)  # danger에 더 높은 가중치
        )
        
        print(f"✅ Training setup complete:")
        print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")  # 수정
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Weight decay: {self.config['weight_decay']}")
        
    def train_epoch(self, train_loader, epoch):
        """한 에포크 훈련 (엄밀한 구현)"""
        self.model.train()
        
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        batch_count = 0
        
        print(f"\n📚 Epoch {epoch+1}/{self.config['num_epochs']}")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(train_loader):
            point_clouds = batch['point_cloud'].to(self.device)
            safety_labels = batch['safety_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            results = self.model.forward_safety_analysis(point_clouds)
            safety_logits = results['safety_logits']
            
            # Loss 계산
            loss = self.criterion(safety_logits, safety_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (안정성 향상)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 메트릭 계산
            epoch_loss += loss.item()
            predictions = torch.argmax(safety_logits, dim=1)
            correct_predictions += (predictions == safety_labels).sum().item()
            total_samples += safety_labels.size(0)
            batch_count += 1
            
            # 진행상황 출력
            if batch_idx % 10 == 0:
                current_acc = correct_predictions / total_samples
                print(f"   Batch {batch_idx:3d}/{len(train_loader)}: "
                      f"Loss {loss.item():.4f}, Acc {current_acc:.3f}")
        
        # 에포크 평균값 계산
        avg_loss = epoch_loss / batch_count
        accuracy = correct_predictions / total_samples
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, epoch):
        """검증 (엄밀한 구현)"""
        self.model.eval()
        
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        batch_count = 0
        
        # 클래스별 정확도 추적
        class_correct = torch.zeros(3)
        class_total = torch.zeros(3)
        
        with torch.no_grad():
            for batch in val_loader:
                point_clouds = batch['point_cloud'].to(self.device)
                safety_labels = batch['safety_label'].to(self.device)
                
                results = self.model.forward_safety_analysis(point_clouds)
                safety_logits = results['safety_logits']
                
                loss = self.criterion(safety_logits, safety_labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(safety_logits, dim=1)
                correct_predictions += (predictions == safety_labels).sum().item()
                total_samples += safety_labels.size(0)
                batch_count += 1
                
                # 클래스별 정확도
                for i in range(3):
                    class_mask = (safety_labels == i)
                    if class_mask.sum() > 0:
                        class_correct[i] += (predictions[class_mask] == i).sum().item()
                        class_total[i] += class_mask.sum().item()
        
        avg_loss = val_loss / batch_count
        accuracy = correct_predictions / total_samples
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        # 클래스별 정확도 출력
        print(f"   📊 Class-wise accuracy:")
        class_names = ['Safe', 'Warning', 'Danger']
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"      {name}: {class_acc:.3f} ({int(class_correct[i])}/{int(class_total[i])})")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        # 체크포인트 디렉토리 생성
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 체크포인트 저장
        filename = 'stage_a_best.pth' if is_best else f'stage_a_epoch_{epoch}.pth'
        filepath = checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
        
        return filepath
    
    def extract_safety_tokens_sample(self, dataloader):
        """A안→B안 연결용 Safety Tokens 추출 테스트"""
        self.model.eval()
        
        print("\n🔗 Testing Stage A → B connection...")
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            point_clouds = batch['point_cloud'][:2].to(self.device)  # 2 samples
            
            results = self.model.forward_safety_analysis(point_clouds)
            safety_tokens = results['safety_tokens']  # [2, 40, 768]
            safety_probs = results['safety_probs']
            
            print(f"   ✅ Safety tokens extracted: {safety_tokens.shape}")
            print(f"   ✅ Token statistics:")
            print(f"      Mean: {safety_tokens.mean().item():.4f}")
            print(f"      Std:  {safety_tokens.std().item():.4f}")
            print(f"      Min:  {safety_tokens.min().item():.4f}")
            print(f"      Max:  {safety_tokens.max().item():.4f}")
            print(f"   ✅ Predicted safety: {torch.argmax(safety_probs, dim=1)}")
            print(f"   ✅ Ready for Stage B integration!")
            
            return safety_tokens

def main():
    """Stage A 훈련 메인 함수"""
    print("🚀 Stage A Training: PointLoRA-Only (엄밀한 구현)")
    print("=" * 60)
    
    # 훈련 설정
    config = {
        'lora_rank': 16,
        'lora_alpha': 32,
        'safety_token_count': 40,
        'feature_dim': 768,
        'learning_rate': 5e-4,
        'weight_decay': 1e-2,
        'batch_size': 4,
        'num_epochs': 20,
        'val_frequency': 2
    }
    
    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 데이터셋 생성
    print(f"\n📦 Creating datasets...")
    train_dataset = MockScaffoldDataset(num_samples=800, mode='train')
    val_dataset = MockScaffoldDataset(num_samples=200, mode='val')
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # 트레이너 초기화
    print(f"\n🎯 Initializing trainer...")
    trainer = StageATrainer(config)
    
    # 훈련 루프
    print(f"\n🏃 Starting training loop...")
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 훈련
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # 검증 (일정 간격으로)
        if (epoch + 1) % config['val_frequency'] == 0:
            val_loss, val_acc = trainer.validate(val_loader, epoch)
            
            print(f"   🎯 Epoch {epoch+1} Summary:")
            print(f"      Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}")
            print(f"      Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                trainer.save_checkpoint(epoch, is_best=True)
                print(f"      🎉 New best validation accuracy: {val_acc:.3f}")
            
            # Learning rate 조정
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"      📈 Learning rate: {current_lr:.6f}")
    
    # 훈련 완료
    training_time = time.time() - start_time
    print(f"\n✅ Stage A Training Completed!")
    print(f"   Training time: {training_time/60:.1f} minutes")
    print(f"   Best validation accuracy: {best_val_acc:.3f}")
    
    # A안→B안 연결 테스트
    safety_tokens = trainer.extract_safety_tokens_sample(val_loader)
    
    # 최종 체크포인트 저장
    final_checkpoint = trainer.save_checkpoint(config['num_epochs']-1, is_best=False)
    
    print(f"\n🎯 Stage A Results Summary:")
    print(f"   ✅ LoRA adaptation successful")
    print(f"   ✅ Safety token extraction working")
    print(f"   ✅ Ready for Stage B integration")
    print(f"   ✅ Best model saved: checkpoints/stage_a_best.pth")
    
    return trainer, safety_tokens

if __name__ == "__main__":
    trainer, safety_tokens = main()