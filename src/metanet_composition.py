"""基于Meta-Net的任务向量动态组合实现

这个模块实现了MetaNet-aTLAS算法，通过Meta-Net网络为每个样本动态生成
任务向量组合系数，实现样本级别的知识组合。

与原始aTLAS不同，MetaNet-aTLAS根据输入样本的特性动态调整不同任务向量的
组合权重，提高模型的适应性和泛化能力。
"""

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers


class MetaNet(nn.Module):
    """Meta-Net网络，用于从样本特征生成任务向量组合系数

    网络结构为两层瓶颈结构(Linear-ReLU-Linear)，输入为样本特征，
    输出为该样本对应的任务向量组合系数。
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        """初始化Meta-Net

        参数:
        ----------
        input_dim: int
            输入特征维度
        output_dim: int
            输出维度，等于任务向量的数量
        hidden_dim: int, optional
            隐藏层维度，默认为输入维度的1/4
        """
        super().__init__()

        # 如果未指定隐藏层维度，则使用输入维度的1/4
        if hidden_dim is None:
            hidden_dim = max(input_dim // 4, output_dim)

        # 两层瓶颈结构：Linear-ReLU-Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化为0，使得初始状态下的组合系数接近零
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        """前向传播

        参数:
        ----------
        x: Tensor [batch_size, input_dim]
            样本特征

        返回:
        ----------
        coefficients: Tensor [batch_size, output_dim]
            每个样本的任务向量组合系数
        """
        return self.net(x)


class MetaNetImageEncoder(nn.Module):
    """基于Meta-Net的图像编码器，用于非线性模型的任务向量动态组合

    接收样本作为输入，先通过基础编码器获取特征，然后通过Meta-Net生成
    任务向量组合系数，最后使用这些系数动态组合任务向量。
    """

    def __init__(self, model, task_vectors, blockwise=False) -> None:
        """初始化Meta-Net图像编码器

        参数:
        ----------
        model: nn.Module
            基础图像编码器模型
        task_vectors: List[NonLinearTaskVector]
            任务向量列表
        blockwise: bool, default: False
            是否对每个参数块使用不同的系数
        """
        super().__init__()

        # 提取基础模型的功能函数和参数
        func, params, self.buffer = make_functional_with_buffers(model)
        # 避免元张量无数据错误
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # 复制图像编码器的属性
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        # 存储任务向量
        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise

        # 获取模型的特征维度，这里假设是最后一个参数对象
        # 注意：不同模型可能需要不同的方式获取特征维度
        if hasattr(model, 'model') and hasattr(model.model, 'ln_final'):
            feat_dim = model.model.ln_final.bias.numel()
        else:
            # 如果无法确定，使用一个默认值或从模型结构推断
            # 这里假设使用ViT的输出维度，通常是768（ViT-B）或1024（ViT-L）
            feat_dim = 768

            # 初始化Meta-Net
        if blockwise:
            # 如果使用blockwise，每个参数块有自己的系数
            self.meta_net = MetaNet(feat_dim, len(task_vectors) * len(self.params))
        else:
            # 否则每个任务向量有一个全局系数
            self.meta_net = MetaNet(feat_dim, len(task_vectors))

    def _apply(self, fn):
        """重写_apply方法以重新定位buffer列表

        注意：此函数签名适用于PyTorch 1.13.1。
        更新版本添加了另一个可选参数`recurse=True`。
        """
        new_self = super()._apply(fn=fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)

    def forward(self, x) -> torch.Tensor:
        """前向传播

        参数:
        ----------
        x: Tensor [batch_size, 3, H, W]
            输入图像批次

        返回:
        ----------
        features: Tensor [batch_size, feature_dim]
            编码后的特征
        """
        # 1. 首先获取基础特征
        with torch.no_grad():
            base_features = self.func(self.params, self.buffer, x)

        # 2. 使用Meta-Net生成组合系数
        if self.blockwise:
            # 如果使用blockwise，将系数重塑为[batch_size, n_task_vectors, n_params]
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params))
        else:
            # 否则系数形状为[batch_size, n_task_vectors]
            batch_coefficients = self.meta_net(base_features)

        # 3. 对批次中的每个样本应用其特定的组合系数
        batch_size = x.size(0)
        all_outputs = []

        for i in range(batch_size):
            if self.blockwise:
                # 对每个参数块使用不同的系数
                coefs = batch_coefficients[i]  # [n_task_vectors, n_params]
                dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]) for i, dp in enumerate(zip(*self.dparams))]
            else:
                # 对每个任务向量使用全局系数
                coefs = batch_coefficients[i]  # [n_task_vectors]
                dparams = [sum([p * c for p, c in zip(dp, coefs)]) for dp in zip(*self.dparams)]

            # 应用组合后的参数差异
            new_params = [dp + p for dp, p in zip(dparams, self.params)]

            # 计算当前样本的输出
            output = self.func(new_params, self.buffer, x[i:i + 1])
            all_outputs.append(output)

        # 4. 合并所有样本的输出
        return torch.cat(all_outputs, dim=0)


class MetaNetLinearizedModel(nn.Module):
    """基于Meta-Net的线性化模型，用于线性化模型的任务向量动态组合

    类似于MetaNetImageEncoder，但专为线性化模型设计。
    """

    def __init__(self, model, task_vectors, blockwise=False) -> None:
        """初始化Meta-Net线性化模型

        参数:
        ----------
        model: LinearizedModel
            线性化的基础模型
        task_vectors: List[LinearizedTaskVector]
            线性化的任务向量列表
        blockwise: bool, default: False
            是否对每个参数块使用不同的系数
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        # 存储任务向量
        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise

        # 获取模型的特征维度
        # 注意：这是一个简化实现，实际中可能需要根据具体模型调整
        feat_dim = 768  # 假设使用ViT-B的特征维度

        # 初始化Meta-Net
        if blockwise:
            self.meta_net = MetaNet(feat_dim, len(task_vectors) * len(self.params0))
        else:
            self.meta_net = MetaNet(feat_dim, len(task_vectors))

    def _apply(self, fn):
        """重写_apply方法以重新定位buffer列表"""
        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        """前向传播

        参数:
        ----------
        x: Tensor [batch_size, 3, H, W]
            输入图像批次

        返回:
        ----------
        outputs: Tensor [batch_size, feature_dim]
            模型输出
        """
        # 1. 首先获取基础特征用于生成系数
        with torch.no_grad():
            base_features = self.func0(self.params0, self.buffers0, x)

        # 2. 使用Meta-Net生成组合系数
        if self.blockwise:
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params0))
        else:
            batch_coefficients = self.meta_net(base_features)

        # 3. 对批次中的每个样本应用其特定的组合系数
        batch_size = x.size(0)
        all_outputs = []

        for i in range(batch_size):
            if self.blockwise:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]) for i, dp in enumerate(zip(*self.dparams))]
            else:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c for p, c in zip(dp, coefs)]) for dp in zip(*self.dparams)]

            # 对线性化模型应用一阶泰勒展开
            out, dp = jvp(
                lambda param: self.func0(param, self.buffers0, x[i:i + 1]),
                (tuple(self.params0),),
                (tuple(dparams),),
            )
            output = out + dp
            all_outputs.append(output)

        # 4. 合并所有样本的输出
        return torch.cat(all_outputs, dim=0)