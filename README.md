# BA-MBRec

BA-MBRec 是一个基于多行为推荐的深度学习模型，旨在通过用户的多种行为（如点击、收藏、加入购物车、购买等）来提升推荐系统的性能。

## 项目结构

```
BP-MBRec/
├── __init__.py
├── graph_utils.py       # 图处理工具，包括数据处理和行为矩阵操作
├── main_pytorch.py      # 主程序，包含模型训练和测试逻辑
├── Model.py             # 模型定义
├── MV_Net.py            # 多视角网络实现
├── Params.py            # 参数配置
├── data/                # 数据目录
│   ├── __init__.py
│   ├── README.md
│   ├── IJCAI_15/        # 数据集文件夹
│   │   ├── meta_multi_single_beh_user_index_shuffle
│   │   ├── trn_buy
│   │   ├── trn_cart
│   │   ├── trn_click
│   │   ├── trn_fav
│   │   └── tst_int
├── Utils/               # 工具目录
│   ├── README.md
│   └── TimeLogger.py    # 时间记录工具
```

## 功能

- **多行为数据处理**：通过 `graph_utils.py` 中的工具函数对用户行为数据进行预处理。
- **深度学习模型**：基于 PyTorch 实现多视角推荐模型。
- **高效训练**：支持负采样和行为矩阵归一化等优化策略。
- **灵活配置**：通过 `Params.py` 文件调整模型参数。

## 数据集
Raw data：

IJCAI contest: https://tianchi.aliyun.com/dataset/dataDetail?dataId=47

Retail Rocket: https://www.kaggle.com/retailrocket/ecommerce-dataset

Tmall: https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

测试项目可以使用 [IJCAI 2015](https://www.ijcai.org/) 提供的多行为推荐数据集，包含以下文件：

- `trn_buy`：训练集中的购买行为
- `trn_cart`：训练集中的加入购物车行为
- `trn_click`：训练集中的点击行为
- `trn_fav`：训练集中的收藏行为
- `tst_int`：测试集中的交互行为
### 环境依赖

请确保已安装以下依赖：

- Python
- PyTorch
- NumPy
- SciPy


### 模型训练

运行主程序以训练模型：

```bash
python main_pytorch.py
```

### 模型评估

训练完成后，模型会在测试集上进行评估，并输出指标结果。

## 主要模块说明

### `graph_utils.py`

- `data_process(test)`：将稀疏矩阵转换为列表形式。
- `get_use(behaviors_data)`：对行为矩阵进行归一化处理并转换为张量。

### `Model.py`

定义了推荐模型的核心结构，包括多视角网络和损失函数。

### `MV_Net.py`

实现了多视角网络（Multi-View Network），用于捕获用户的多种行为模式。

### `Params.py`

包含模型的超参数配置，例如学习率、批量大小等。

## 贡献

欢迎对本项目提出建议或贡献代码！请通过提交 Issue 或 Pull Request 与我们联系。

## 参考文献
有关 BA-MBRec 的详细信息，请参考以下论文：
Behavior-Type Aware Representation Learning for Multiplex Behavior Recommendation
Anonymous Author(s), Anonymous Institution
Available at: https://github.com/sunshixx/BA-MBRec/tree/master 

## 许可证

本项目基于 MIT 许可证开源，详情请参阅 [LICENSE](LICENSE) 文件。
