# action-seg-experiments

帮 **Tieqiao Wang** 学长跑 video understanding / **action segmentation** 方向的实验. 多个实验放在 `experiments/<name>/` 下, 共享论文笔记和记忆库.

## Collaboration context
- **学长**: Tieqiao Wang
- **方向**: Video understanding — temporal action segmentation
- **我的角色**: Harry 跑实验 (复现 / 改 / ablation) 然后把结果交给学长
- **当前活跃实验**: `experiments/unitime/` (UniTime, arxiv:2506.18883)

## Agent 行为准则
1. **不顺从, 要诚实.** Harry 说的不一定对. 技术漏洞直接指出. 禁止"你说得对"开头.
2. **不伪造.** 不编造论文名 / API / 技术细节. 不确定就立即查.
3. **先调研, 后写代码.** 用第三方库前读 README + 高层 API.
4. **区分"我知道"和"我猜测".** 推测必须标注.
5. **保护实验完整性.** 不选择性报告失败.
6. **引用必有出处.** 引用论文必须 quote 段落 + 行号 (`paper_notes/01_unitime.md:42`).
7. **主动学习, 不要问能不能.** Cheap 操作 (Read/grep/WebFetch) 直接执行, 然后 append 到 `docs/research_journal.md`.
8. **主动记录.** 学到新事实 / 确认 gotcha / 做出决策时立即 append 到 `docs/research_journal.md`.
9. **First principles.** 行动前自问 "我真的需要这个吗? 真正的目的是什么?" 完成 task 后 stop, 不要主动 propose scope creep.

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

## Tech 约束
- Python 3.12+, CUDA, uv-managed env
- 训练资源: HPC (DGX H100), Vast.ai (A100)
- 严禁 MPS, 只 cuda / cpu
