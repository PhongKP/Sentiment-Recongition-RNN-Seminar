from data import train_loader, test_loader, vocab
from models import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01):
    # Khởi tạo loss function và optimizer SGD (không dùng Adam)
    # [Sinh viên bổ sung: dùng CrossEntropyLoss và optim.SGD]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for text, labels in train_loader:
            # [Sinh viên bổ sung: forward, tính loss, backward, cập nhật trọng số bằng SGD]
            # Xóa gradient cũ
            optimizer.zero_grad()
            # Forward pass
            predictions = model(text)
            # Tính loss
            loss = criterion(predictions, labels)
            # Backward pass và cập nhật trọng số
            loss.backward()
            optimizer.step()

            # Cập nhật loss tổng
            epoch_loss += loss.item()
            batch_count += 1
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
   
    # Đánh giá mô hình
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, labels in test_loader:
            # [Sinh viên bổ sung: dự đoán và thu thập kết quả]
            # Forward pass
            outputs = model(text)

            # Lấy nhãn dự đoán (nhãn có xác suất cao nhất)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

# Thử nghiệm Pretrained vs Scratch
results = {}
for pretrained in [True, False]:
    print(f"Huấn luyện mô hình với pretrained={pretrained}")
    model = RNNModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=pretrained)
    key = f"RNN_Pretrained={pretrained}"
    acc, f1 = train_and_evaluate(model, train_loader, test_loader)
    results[key] = {"Accuracy": acc, "F1-score": f1}
    print(f"{key} - Accuracy: {acc}, F1-score: {f1}")
    
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Kết quả đã được lưu vào results.json")