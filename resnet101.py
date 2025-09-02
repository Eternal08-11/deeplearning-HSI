import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import random
import time
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# 配置运行环境（确保普通电脑可运行）
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用CPU
# torch.set_num_threads(2)  # 限制CPU线程数

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 检查GPU是否可用
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ----------------------------
# 数据加载与预处理
# ----------------------------
def load_data(data_path, label_path):
    """加载并预处理数据"""
    try:
        # 加载数据
        hyperspectral_data = np.load(data_path)
        hyperspectral_data = np.transpose(hyperspectral_data, (1 ,2, 0))  # 调整为 (行, 列, 波段)
        print(f"原始数据形状: {hyperspectral_data.shape}")
        labels = np.load(label_path)
        
        # 展平数据
        rows, cols, bands = hyperspectral_data.shape
        X = hyperspectral_data.reshape(-1, bands)
        Y = labels.flatten()
        
        # 过滤无效样本（标签为0的样本）
        valid_mask = Y != 0
        X = X[valid_mask]
        Y = Y[valid_mask].astype(int)
        
        # 获取唯一类别并排序
        unique_classes = np.unique(Y)
        print(f"数据加载完成: {X.shape[0]}样本, {bands}波段, {len(unique_classes)}类别: {unique_classes}")
        return X, Y, bands, unique_classes
        
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        return None, None, 0, None


def prepare_dataset(X, Y, unique_classes, max_samples=76257, train_ratio=0.7):
    """准备训练集和测试集"""
    # 限制样本数量以提高速度
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        print(f"样本量过大，已随机采样至{max_samples}个")
    
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1-train_ratio, random_state=42, stratify=Y
    )
    
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 保留前3750个波段
    if X_train.shape[1] >= 3750:
        X_train = X_train[:, :3750]
        X_test = X_test[:, :3750]
        print(f"已保留前3750个波段")
    else:
        print(f"特征数不足3750，全部保留")

    # 标签从0开始编码（使用原始类别信息确保一致性）
    class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
    Y_train = np.array([class_mapping[cls] for cls in Y_train])
    Y_test = np.array([class_mapping[cls] for cls in Y_test])

    return X_train, X_test, Y_train, Y_test


# ----------------------------
# ResNet模型
# ----------------------------
class ResNetModel(nn.Module):
    def __init__(self, num_classes, in_channels, model_type='resnet50'):
        super(ResNetModel, self).__init__()
        resnet_dict = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }
        if model_type not in resnet_dict:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.resnet = resnet_dict[model_type](weights=None)
        # 替换输入层以适应光谱数据的输入通道数
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 替换最后的全连接层以适应类别数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # 将输入调整为ResNet期望的形状 [batch, channels, height, width]
        x = x.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 3)  # 重复3次以模拟RGB通道
        return self.resnet(x)


# ----------------------------
# 训练函数
# ----------------------------
def train_cnn(X_train, Y_train, X_test, Y_test, num_classes, model_type='resnet50',
              epochs=100, batch_size=64, lr=0.001):
    """训练CNN模型"""
    # 转换为PyTorch张量并移动到设备
    X_train = torch.FloatTensor(X_train).to(device)
    Y_train = torch.LongTensor(Y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    Y_test = torch.LongTensor(Y_test).to(device)
    
    # 处理类别不平衡（加权采样）
    class_counts = np.bincount(Y_train.cpu().numpy())
    weights = (1.0 / torch.FloatTensor(class_counts)).to(device)
    sampler = WeightedRandomSampler(
        weights=weights[Y_train],
        num_samples=len(Y_train),
        replacement=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False  # 使用sampler时关闭shuffle
    )
    
    # 初始化模型和优化器
    model = ResNetModel(num_classes, X_train.shape[1], model_type).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,  # 添加动量以加速收敛
        weight_decay=1e-4  # 轻微正则化防止过拟合
    )
    
    # 使用ReduceLROnPlateau动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 启用混合精度训练
    scaler = torch.amp.GradScaler()
    
    # 训练跟踪
    best_acc = 0.0
    history = []
    
    print(f"\n开始训练 {model_type}（{epochs}轮，{batch_size}批次）")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 训练批次
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            
            # 将张量从 GPU 移动到 CPU
            predicted = predicted.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()
            
            val_acc = accuracy_score(Y_test_cpu, predicted) * 100
        
        # 记录最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        
        # 更新学习率
        scheduler.step(train_loss)
        history.append((train_loss, val_acc))
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n训练完成！耗时: {total_time:.2f}秒 | 最佳准确率: {best_acc:.2f}%")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
    return model, best_acc, history


# ----------------------------
# 评估与可视化
# ----------------------------
def evaluate_model(model, X_test, Y_test, unique_classes, save_plot=True):
    """评估模型并可视化结果"""
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test).to(device))
        _, predicted = torch.max(outputs, 1)
    
    # 如果 Y_test 是 PyTorch 张量，则移动到 CPU，否则直接使用
    if isinstance(Y_test, torch.Tensor):
        Y_test = Y_test.cpu().numpy()
    predicted = predicted.cpu().numpy()  # 确保 predicted 是 NumPy 数组
    
    # 计算准确率
    acc = accuracy_score(Y_test, predicted) * 100
    print(f"\n最终测试准确率: {acc:.2f}%")
    
    # 打印分类报告
    print("\n分类报告:")
    class_names = [f'类别{i}' for i in unique_classes]
    print(classification_report(Y_test, predicted, labels=range(len(unique_classes)), 
                                target_names=class_names))
    
    # 绘制混淆矩阵
    if save_plot:
        cm = confusion_matrix(Y_test, predicted)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测类别')
        plt.ylabel('实际类别')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("混淆矩阵已保存为confusion_matrix.png")
    
    return acc


# ----------------------------
# 参数调优
# ----------------------------
def train_and_evaluate(X_train, Y_train, X_test, Y_test, unique_classes, model_types, param_grid):
    best_model = None
    best_acc = 0.0
    best_params = None

    for model_type in model_types:
        for params in param_grid:
            print(f"\n正在训练模型: {model_type}，参数: {params}")
            model, acc, history = train_cnn(
                X_train, Y_train, X_test, Y_test,
                num_classes=len(unique_classes),
                model_type=model_type,
                **params
            )
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {'model_type': model_type, **params}
                torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            print(f"当前最佳准确率: {best_acc:.2f}%")

    print(f"\n最佳模型: {best_params}，准确率: {best_acc:.2f}%")
    return best_model, best_params


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 设置随机种子确保可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset_folder = 'D:\\HSI\\new dataset'
    max_samples = 50000
    epochs = 100
    batch_size = 64

    # 查找数据文件
    try:
        data_files = [f for f in os.listdir(dataset_folder) if 'data' in f.lower() and f.endswith('.npy')]
        label_files = [f for f in os.listdir(dataset_folder) if 'label' in f.lower() and f.endswith('.npy')]

        if not data_files or not label_files:
            print("未找到数据或标签文件")
            return

        # 使用第一个数据文件对
        data_path = os.path.join(dataset_folder, data_files[0])
        label_path = os.path.join(dataset_folder, label_files[0])

        # 1. 加载数据 - 获取唯一类别信息
        X, Y, bands, unique_classes = load_data(data_path, label_path)
        if X is None or unique_classes is None:
            return

        num_classes = len(unique_classes)
        if num_classes < 2:
            print("有效类别数不足，无法训练")
            return

        print(f"检测到{num_classes}个类别，已生成对应类别名称")

        # 2. 准备数据集 - 使用唯一类别信息进行标签映射
        X_train, X_test, Y_train, Y_test = prepare_dataset(
            X, Y, unique_classes, max_samples=max_samples
        )

        # 可视化类别分布
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.hist(Y_train, bins=len(unique_classes), rwidth=0.8)
        plt.title('训练集类别分布')
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.subplot(1,2,2)
        plt.hist(Y_test, bins=len(unique_classes), rwidth=0.8, color='orange')
        plt.title('测试集类别分布')
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        print('类别分布已保存为class_distribution.png')

        # 定义不同尺寸的 ResNet 模型
        model_types = [ 'resnet101']

        # 训练和评估每个模型
        for model_type in model_types:
            print(f"\n开始训练模型: {model_type}")
            model, best_acc, history = train_cnn(
                X_train, Y_train, X_test, Y_test,
                num_classes=num_classes,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size
            )

            # 评估模型 - 传递原始唯一类别信息
            evaluate_model(model, X_test, Y_test, unique_classes)

            # 绘制训练曲线
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot([h[0] for h in history], label='训练损失')
            plt.title(f'{model_type} 训练损失曲线')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot([h[1] for h in history], label='验证准确率', color='orange')
            plt.title(f'{model_type} 验证准确率曲线')
            plt.xlabel('轮次')
            plt.ylabel('准确率 (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'training_curve_{model_type}.png')
            print(f"{model_type} 训练曲线已保存为training_curve_{model_type}.png")

            # 绘制训练损失曲线
            plt.figure(figsize=(8, 6))
            plt.plot([h[0] for h in history], label='训练损失')
            plt.title(f'{model_type} 训练损失曲线')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.grid()
            plt.savefig(f'loss_curve_{model_type}.png')
            plt.close()
            print(f"{model_type} 训练损失曲线已保存为loss_curve_{model_type}.png")

    except Exception as e:
        print(f"运行错误: {str(e)}")


if __name__ == '__main__':
    main()
