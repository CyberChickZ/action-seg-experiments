# action-seg-experiments

帮 **Tieqiao Wang** 学长跑 video understanding / **action segmentation** 方向的实验. 多个实验放在 `experiments/<name>/` 下, 共享论文笔记和记忆库.

## Collaboration context
- **学长**: Tieqiao Wang
- **方向**: Video understanding — temporal action segmentation
- **我的角色**: Harry 跑实验 (复现 / 改 / ablation) 然后把结果交给学长
- **当前活跃实验**: `experiments/unitime/` (UniTime, arxiv:2506.18883)

## 4 层记忆系统

```
L1 Sources    : paper_notes/*.md            (Harry 读论文写的标注)
L2 Synthesis  : docs/research_*.md          (Harry 的思考, Claude 也写 research_journal.md)
L3 Trace      : daily/YYYY-MM-DD.md         (SessionEnd hook 自动 capture 对话)
L4 Compiled   : knowledge/concepts/, ...    (LLM 编译产物, knowledge/index.md 是入口)
```

每次 SessionStart hook 自动注入 L1+L2+L3+L4 最新切片. 注入只是指针, **Read 全文是必须的**.

### 必读 (新对话强制 reading order)
1. `CLAUDE.md` (本文件, 自动加载)
2. `docs/research_journal.md` — Harry 的 insights + 决策 (核心)
3. `knowledge/index.md` — 编译后的概念索引
4. `experiments/<active>/README.md` — 当前实验 context
5. `paper_notes/*.md` — 引用任何论文前必须 Read 对应文件

### 记忆操作 slash commands
- `/memory:flush` — 强制把当前对话提取到今天的 daily log
- `/memory:compile` — 把 daily logs 编译成 knowledge concepts
- `/memory:compile --file paper_notes/01_unitime.md` — 单独编译某个 doc 文件
- `/memory:query "..."` — index-guided 查询知识库
- `/memory:lint` — 知识库健康检查

## Experiments
| Dir | Paper | Status |
|-----|-------|--------|
| `experiments/unitime/` | [UniTime](paper_notes/01_unitime.md) — Universal VTG with MLLMs | 待 Harry 读论文 + 启动 |

## HPC 环境规范 (必须遵守)

### bashrc alias
HPC 上每个 conda env 都有对应的 bashrc alias（如 `unitime-gemma4`），在 noVNC terminal 里输入 alias 名即可激活完整环境。**写运行命令时不需要手动 export，只需先调用 alias。**

已有 alias:
- `unitime-gemma4` — UniTime + Gemma4/Qwen3-VL 实验环境

### HF 模型加载规范
- **必须** `export HF_HOME=/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache`（alias 里已含）
- 代码中用 HF model ID 加载（如 `Qwen/Qwen3-VL-2B-Instruct`），**不要硬编码本地路径**
- 需要 `cache_dir` 参数时传 `HF_HOME` 环境变量值
- 下载新模型：让 `from_pretrained` 自动下载到 HF cache，或 `huggingface-cli download`

### 三层存储
| 层 | 路径 | 容量 | 用途 |
|---|---|---|---|
| home | `/nfs/stak/users/zhanhaoc/` | 25GB | 只放 .bashrc, symlinks |
| exascale | `/nfs/hpc/share/zhanhaoc/` | 1.5T | 代码, conda env |
| dgx2-2 | `/nfs/hpc/dgx2-2/zhanhaoc/` | 25T | 模型权重, HF cache, 数据集 |

**大文件（模型、数据集、cache）只能放 dgx2-2。**

## Tech 约束
- Python 3.12+, CUDA, uv-managed env
- 训练资源: HPC (DGX H100), Vast.ai (A100)
- 严禁 MPS, 只 cuda / cpu
