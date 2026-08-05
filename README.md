# vie-plugin-indicator-light

指示灯状态比对插件。插件检测指示灯 ROI，通过动态 batch 特征模型一次生成全部
embedding，再按注册灯位布局建立几何对应关系并比较状态。正常布局使用轻量相似
变换匹配；布局异常时使用 ORB/RANSAC 图像配准兜底。

## API

- 路径：`POST /api/v1/indicator_light_detect`
- 表单字段：`file` 为当前图，`json_data` 为 JSON 字符串
- `type`：物料号，也是回流数据目录键
- `modelParams.type`：用于匹配 `AICameraModel.Version`
- `AICameraModel[].ModelFile`：注册参考图地址

注册模式字段在 JSON 中仍使用 `modelParams.register`，当前仅保留契约。

## 模型

| 模型 | 默认路径 | 要求 |
| --- | --- | --- |
| ROI 检测 | `./weights/indicator_light/rfdetr-small.onnx` | 单类 RF-DETR 检测，固定输入 `1×3×768×768` |
| 特征识别 | `./weights/indicator_light/rec_v3.onnx` | 输入和输出必须支持动态 batch |

固定 batch 的旧识别模型会在启动时被拒绝。全部 ROI 在一次 runner 调用中完成识别。

## 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `INDICATOR_VECTOR_CACHE_ENABLED` | `true` | 是否启用 Chroma 注册向量缓存 |
| `INDICATOR_VECTOR_CACHE_PATH` | `./data/indicator_light/chroma` | 持久化目录 |
| `INDICATOR_VECTOR_COLLECTION` | `registered_embeddings` | collection 名称 |
| `INDICATOR_DOWNLOAD_CONNECT_TIMEOUT` | `3` | 注册图连接超时秒数 |
| `INDICATOR_DOWNLOAD_READ_TIMEOUT` | `10` | 注册图读取超时秒数 |
| `INDICATOR_MAX_REGISTERED_IMAGE_MB` | `20` | 注册图最大体积 |
| `INDICATOR_ALLOWED_HOSTS` | 空 | 允许下载的主机，逗号分隔 |

缓存同时保存注册 embedding、归一化检测框、图像宽高和布局版本。旧格式缓存会在
首次读取时失效并重新生成。缓存初始化失败会降级为无缓存模式；模型指纹失败会
拒绝场景初始化。

待测图中无法匹配到任何注册灯位的额外候选框会记录日志并忽略；注册灯位缺失、
布局映射歧义或异常配准失败时返回 `unmatch`，不会回退到按索引强制比较。

## 安装、示例与测试

在框架仓库根目录执行：

```bash
conda run -n mobile_vision pip install -e plugins/vie-plugin-indicator-light --no-deps
conda run -n mobile_vision python plugins/vie-plugin-indicator-light/examples/run.py \
  /path/to/current.jpg /path/to/registered.jpg
```

在本插件目录执行测试：

```bash
conda run -n mobile_vision env PYTHONPATH=../..:. python -m pytest tests/ -v
```

插件通过 `ScenarioRegistry` 创建场景实例，两个 runner 会在初始化失败或服务关闭时释放。
变更记录见 [CHANGELOG.md](CHANGELOG.md)。
