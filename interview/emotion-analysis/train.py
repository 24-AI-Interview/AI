# ipynb 파일로 학습을 완료한 후 깃허브에 올리기 위해 py 파일로 변환한 코드
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from facenet_pytorch import InceptionResnetV1
import os

# ==========================================
# 1. 기본 설정 및 하이퍼파라미터
# ==========================================
DATA_DIR = '/kaggle/input/dataset/dataset' 
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 학습 장치: {DEVICE}")

# ==========================================
# 2. 데이터 증강 (Augmentation) 및 로더 준비
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform: x = self.transform(x)
            return x, y
        def __len__(self): return len(self.subset)

    train_dataset = CustomDataset(train_subset, transform=train_transforms)
    val_dataset = CustomDataset(val_subset, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"데이터 준비 완료! (학습: {len(train_dataset)}장 / 검증: {len(val_dataset)}장)")

except Exception as e:
    print(f"오류: 경로를 확인해주세요! ({DATA_DIR})")
    raise e

# ==========================================
# 3. 공통 학습 루프 함수 (중복 코드 제거)
# ==========================================
def run_training_phase(model, phase_name, epochs, optimizer, scheduler, criterion, best_acc=0.0):
    print(f"\n{'='*50}")
    print(f"🔥 {phase_name} 시작 (총 {epochs} Epochs)")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        
        # 검증 (Validation)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        
        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[{phase_name} - Epoch {epoch+1:02d}] LR: {current_lr:.8f} | "
              f"Train: {epoch_acc:.2f}% (Loss {epoch_loss:.4f}) | "
              f"Val: {val_epoch_acc:.2f}% (Loss {val_epoch_loss:.4f})")
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            save_name = f"best_model_{phase_name.replace(' ', '_')}.pt"
            torch.save(model.state_dict(), save_name)
            print(f"  --> 🏆 최고 기록 갱신! ({best_acc:.2f}%) '{save_name}' 저장됨")
            
    return best_acc

# ==========================================
# 4. 메인 실행 부 (4단계 점진적 학습)
# ==========================================
if __name__ == '__main__':
    # 모델 초기화
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    global_best_acc = 0.0

    # ---------------------------------------------------------
    # [1단계] 완전 봉쇄 (Classifier Only)
    # ---------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.logits.parameters():
        param.requires_grad = True
    for param in model.last_linear.parameters():
        param.requires_grad = True

    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.01)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=3)
    
    global_best_acc = run_training_phase(model, "Phase 1_Classifier Only", 25, optimizer1, scheduler1, criterion, global_best_acc)

    # ---------------------------------------------------------
    # [2단계] 부분 개방 (Mixed_7a 잠금 해제)
    # ---------------------------------------------------------

    model.load_state_dict(torch.load('best_model_Phase_1_Classifier_Only.pt'))
    
    for name, param in model.named_parameters():
        if "Mixed_7a" in name or "last_linear" in name or "logits" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.01)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=3)
    
    global_best_acc = run_training_phase(model, "Phase 2_Mixed_7a", 15, optimizer2, scheduler2, criterion, global_best_acc)

    # ---------------------------------------------------------
    # [3단계] 허리까지 해제 (Mixed_6a부터 끝까지)
    # ---------------------------------------------------------
    model.load_state_dict(torch.load('best_model_Phase_2_Mixed_7a.pt'))
    
    # Mixed_6a, Block17, Mixed_7a, Block8, last_linear, logits 포함
    unfreeze_blocks = ["Mixed_6a", "Block17", "Mixed_7a", "Block8", "last_linear", "logits"]
    for name, param in model.named_parameters():
        if any(block in name for block in unfreeze_blocks):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer3 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.01)
    scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.1, patience=3)
    
    global_best_acc = run_training_phase(model, "Phase 3_Deep Unfreeze", 20, optimizer3, scheduler3, criterion, global_best_acc)

    # ---------------------------------------------------------
    # [4단계] 쥐어짜기 (Full Unfreeze & Strong Regularization)
    # ---------------------------------------------------------
    model.load_state_dict(torch.load('best_model_Phase_3_Deep_Unfreeze.pt'))
    
    for param in model.parameters():
        param.requires_grad = True

    # 극단적으로 낮은 학습률과 강한 L2 규제
    optimizer4 = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.2)
    scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='min', factor=0.5, patience=1)
    
    final_acc = run_training_phase(model, "Phase 4_Final Squeeze", 10, optimizer4, scheduler4, criterion, global_best_acc)

    print(f"\n모든 학습 공정이 완료되었습니다! 최종 최고 정확도: {final_acc:.2f}%")