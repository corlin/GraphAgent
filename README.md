# GraphAgent 项目文档

## 概述

GraphAgent 是一个基于 GraphRAG 框架构建的项目，旨在处理和查询与《红楼梦》相关的文本数据。该项目通过自定义的下载工具收集《红楼梦》的内容，并利用 GraphRAG 的强大功能构建知识图谱，进行实体提取、社区报告生成和复杂的查询操作。

## 功能

- **数据收集**：使用 `HLM_Downloader.py` 脚本从指定网站下载《红楼梦》的文本内容。
- **数据处理**：通过 GraphRAG 框架对文本数据进行分块、嵌入和存储。
- **实体提取**：识别文本中的组织、人物、地点和事件等实体。
- **社区报告**：生成关于文本数据的社区结构报告。
- **查询功能**：支持本地搜索、全局搜索和基本搜索等多种查询方式。

## 安装

要安装和设置 GraphAgent 项目，请按照以下步骤操作：

```shell
# 安装 GraphRAG
pip install graphrag

# 初始化项目
graphrag init --root .

# 索引数据
graphrag index --root .
```

## 使用方法

### 数据下载

运行 `HLM_Downloader.py` 脚本以从 `http://www.purepen.com/hlm/` 下载《红楼梦》的文本内容：

```shell
python HLM_Downloader.py
```

下载的内容将保存在 `output` 目录下。

### 查询数据

使用 GraphRAG 的查询功能来搜索与《红楼梦》相关的内容，例如测绘合同相关条款：

```shell
graphrag query --root . --method local --query "描述故事梗概" --verbose --dry-run
```

您可以根据需要调整参数：
- `--dry-run` / `--no-dry-run`：是否执行实际查询。
- `--show-completion`：显示查询完成情况。

## 项目结构

- **input/**：存储输入数据文件的目录。
- **output/**：存储处理后的输出数据和下载的《红楼梦》文本内容的目录。
- **cache/**：用于存储缓存数据的目录。
- **logs/**：存储日志文件的目录。
- **prompts/**：包含用于实体提取、描述总结、社区报告和查询的自定义提示文件。
- **HLM_Downloader.py**：用于从网站下载《红楼梦》文本内容的脚本。
- **settings.yaml**：GraphRAG 框架的配置文件，包含模型设置、输入设置、存储设置和工作流程设置等。

## 配置文件

`settings.yaml` 文件包含了 GraphRAG 框架的所有配置选项，包括使用的语言模型（LLM）、嵌入模型、输入数据路径、存储路径以及各种工作流程的设置。可以通过修改此文件来定制项目的行为。

## 贡献

欢迎对 GraphAgent 项目进行贡献。如果您有任何改进建议或发现了问题，请提交 issue 或 pull request。

## 许可证

本项目遵循 MIT 许可证。详情请参见 LICENSE 文件。

## 联系方式

如有任何问题或需要进一步的信息，请联系项目维护者。
