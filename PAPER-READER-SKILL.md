# 📖 Paper Reading 完整工作流 Skills

> 本文档包含论文批读的完整工作流所需的两个 Skill：
> 1. **MinerU** — PDF 解析为 Markdown
> 2. **Paper Reader** — 按 Section 拆分 + 批读批注
>
> 工作流：`arXiv/PDF URL` → MinerU 解析 → Paper Reader 批读 → GitHub Push

---

# Part 1: 📄 MinerU - 文档解析

**OpenDataLab 出品**

> PDF/Word/PPT/图片 → 结构化 Markdown，公式表格全保留！

---

## 🔗 资源链接

| 资源 | 链接 |
|------|------|
| **官网** | https://mineru.net/ |
| **API 文档** | https://mineru.net/apiManage/docs |
| **GitHub** | https://github.com/opendatalab/MinerU |

---

## 🎯 功能

### 支持的文件类型

| 类型 | 格式 |
|------|------|
| 📕 **PDF** | 论文、书籍、扫描件 |
| 📝 **Word** | .docx |
| 📊 **PPT** | .pptx |
| 🖼️ **图片** | .jpg, .png (OCR) |

### 核心优势

1. **公式完美保留** - LaTeX 格式输出
2. **表格结构识别** - 复杂表格也能搞定
3. **多语言 OCR** - 中英文混排无压力
4. **版面分析** - 多栏、图文混排自动处理

---

## 🚀 API 使用 (v4)

### 认证

```bash
# Header 认证
Authorization: Bearer {YOUR_API_KEY}
```

### 单文件解析

```bash
# 1. 提交任务
curl -X POST "https://mineru.net/api/v4/extract/task" \
  -H "Authorization: Bearer $MINERU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://arxiv.org/pdf/2410.17247",
    "enable_formula": true,
    "enable_table": true,
    "layout_model": "doclayout_yolo",
    "language": "en"
  }'

# 返回: {"task_id": "xxx", "status": "pending"}

# 2. 轮询结果
curl "https://mineru.net/api/v4/extract/task/{task_id}" \
  -H "Authorization: Bearer $MINERU_TOKEN"

# 返回: {"status": "done", "result": {...}}
```

### 批量解析

```bash
# 1. 获取上传 URL
curl -X POST "https://mineru.net/api/v4/file-urls/batch" \
  -H "Authorization: Bearer $MINERU_TOKEN" \
  -d '{"file_names": ["paper1.pdf", "paper2.pdf"]}'

# 2. 上传文件到返回的 presigned URLs

# 3. 批量提交任务
curl -X POST "https://mineru.net/api/v4/extract/task/batch" \
  -H "Authorization: Bearer $MINERU_TOKEN" \
  -d '{"files": [{"url": "...", "name": "paper1.pdf"}, ...]}'
```

---

## ⚙️ 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `url` | string | 文件 URL (支持 http/https) |
| `enable_formula` | bool | 启用公式识别 (默认 true) |
| `enable_table` | bool | 启用表格识别 (默认 true) |
| `layout_model` | string | `doclayout_yolo` (快) / `layoutlmv3` (准) |
| `language` | string | `en` / `ch` / `auto` |
| `model_version` | string | `pipeline` / `vlm` / `MinerU-HTML` |

### 模型版本对比

| 版本 | 速度 | 准确度 | 适用场景 |
|------|------|--------|----------|
| `pipeline` | ⚡ 快 | 高 | 常规文档 |
| `vlm` | 🐢 慢 | 最高 | 复杂版面 |
| `MinerU-HTML` | ⚡ 快 | 高 | 网页样式输出 |

---

## 📂 输出结构

解析完成后下载的 ZIP 包含：

```
output/
├── full.md           # 完整 Markdown
├── content_list.json # 结构化内容
├── images/           # 提取的图片
└── layout.json       # 版面分析结果
```

---

## 🔧 OpenClaw 集成工作流

### 论文解析流程

```bash
# 1. 创建论文目录
mkdir -p "./paper-reading/[CVPR 2025] NewPaper"
cd "./paper-reading/[CVPR 2025] NewPaper"

# 2. 提交解析任务
TASK_ID=$(curl -s -X POST "https://mineru.net/api/v4/extract/task" \
  -H "Authorization: Bearer $MINERU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/XXXX.XXXXX"}' | jq -r '.task_id')

# 3. 等待完成 & 下载
# (轮询 status 直到 done，然后下载 result.zip)

# 4. 解压
unzip result.zip -d .
```

### 环境变量

在 `~/.bashrc` 或 OpenClaw config 中设置：

```bash
export MINERU_TOKEN="your_api_key_here"
```

---

## ⚠️ 限制

| 限制 | 数值 |
|------|------|
| 单文件大小 | 200 MB |
| 单文件页数 | 600 页 |
| 并发任务数 | 根据套餐 |

---

## 💡 使用技巧

1. **arXiv 论文直接用 URL**：`https://arxiv.org/pdf/2410.17247`
2. **中文论文用 `language: ch`**
3. **复杂表格用 `vlm` 模型**
4. **批量处理省 quota**：一次提交多个文件更高效

---
---

# Part 2: 📖 Paper Reader - 论文批读工作流

将 MinerU 解析的 `full.md` 自动拆分成按 Section 组织的笔记文件，使用**批读格式**：原文中内嵌批注。

---

## 🎯 功能概览

```
输入: MinerU 解析后的论文目录
      └── full.md + images/ + content_list.json

输出: 完整的批读笔记
      ├── README.md                (论文概览 + Section 导航)
      └── sections/
          ├── 00-abstract.md       (原文 + 图片 + 内嵌批注)
          ├── 01-introduction.md
          ├── 02-related-work.md
          └── ...
```

---

## ⚠️ 强制规则（必须遵守）

### 规则 1：原文必须完整保留

批读 ≠ 改写。Section 文件中的原文必须**逐字从 full.md 复制**，不得省略、压缩或改写任何段落。

- **逐段复制** full.md 对应 section 的所有文字，一字不改
- **批注是额外添加的**，插在原文段落之间，用 `> 💡` 标记
- 完成后**必须 QA**：对比 full.md 行数，确保无遗漏

❌ 错误做法：把原文概括成几句话，用批注替代原文
✅ 正确做法：原文完整复制 → 段落之间插入批注解释

---

### 规则 2：清理 MinerU 解析产生的排版问题

MinerU 从 PDF 提取文本时会引入排版问题，必须修复：

1. **标题文字不要重复**：MinerU 经常在 subsection 标题下面重复标题文字（如 `## 2.1 MLLM for Planning` 下面又出现 `MLLM for Planning Existing studies...`），删掉重复部分
2. **被图片切断的段落要合并**：PDF 排版中图片经常插在段落中间，导致 full.md 里一个完整段落被 Figure 切成两半（如引用 `[6-` 和 `8, 37]` 被 Figure 分开），需要手动合并段落并把 Figure 移到合适位置
3. **清理无法渲染的 LaTeX 宏**：GitHub 不支持 `\operatorname`、`\mathrm`、`\mathbf`、`\mathsf`、`\mathbb` 等宏，作者名和简单文本改成纯文本，复杂公式用图片
4. **跨 Section 的 Figure 放到合适位置**：MinerU 按 PDF 页面顺序提取，但图片实际归属可能不同。遵循以下原则：
   - **Abstract 不放图**：Abstract 页面的 Figure（通常是 Figure 1）归属于 Introduction
   - **Related Work 不放图**：Related Work 页面的 Figure 归属于它实际描述的 Section（通常是 Method 或 Dataset）
   - 根据 Figure 的 caption 内容判断它属于哪个 Section，放到对应位置

---

### 规则 3：图片和表格必须用 MinerU 原始图片

**Figure 和 Table 都必须用 images/ 目录中的原始图片**，不要用 full.md 里的 Markdown/HTML 表格。

1. 查看 `content_list.json`，找到所有 type=image 和 type=table 的条目及其 `img_path`
2. 使用相对路径引用：`![Figure/Table N](../images/xxx.jpg)`
3. 在图片下方添加斜体 caption：`*Figure/Table N: 描述*`
4. **Table 优先用图片**而非 Markdown 表格（论文原始排版更易读）
5. 每个 Figure/Table 后面加批读批注：`> 💡 **Figure/Table N 批读**: ...`

---

### 规则 4：大公式用图片（不要用 LaTeX）

GitHub 无法正确渲染复杂 LaTeX 公式。**所有 display math（`$$...$$`）都用图片替换。**

- **行内简单公式**（如 `$x_i$`、`$N$`）可以保留
- **Display math / 复杂公式**用图片：
  1. 首先检查 `content_list.json` 或 `content_list_v2.json`：找 `type=equation_interline` 的条目
  2. 如果有公式图片：`![Equation](../images/xxx.jpg)`
  3. 如果没有：从 PDF 用 `pymupdf` 截取

---

### 规则 5：多子图 Figure 必须用完整整图

MinerU 经常把一个多子图 Figure 拆成多张独立小图。**必须放完整整图。**

#### 检测方法
在 `full.md` 中，如果**连续多个 `![](images/xxx.jpg)` 共享同一个 Figure caption**，说明被拆分了。

#### 修复方法
从原始 PDF 用 `pymupdf` 提取完整 Figure：
```python
import fitz
doc = fitz.open('paper.pdf')
zoom = 3.0
mat = fitz.Matrix(zoom, zoom)
page = doc[page_index]
clip = fitz.Rect(left, top, right, bottom)
pix = page.get_pixmap(matrix=mat, clip=clip)
pix.save('images/figureN_full.jpg')
```

| 情况 | 做法 |
|------|------|
| 1 张图 + 1 个 caption | ✅ 直接用 MinerU 的图 |
| N 张图 + 1 个 caption (N≥2) | ⚠️ 从 PDF 提取整图 |
| Figure 有 (a)(b)(c) 子图标签 | ⚠️ 大概率被拆，检查 |

---

### 规则 6：每篇论文必须有 README.md

每个论文目录下必须有 `README.md` 作为入口页面。

内容要求：
1. **论文元信息**：标题、作者、会议/期刊、年份、链接
2. **一句话总结**
3. **核心贡献**：3-5 点
4. **Section 导航表**：链接到每个 section 文件，带简短描述
5. **关键数字速查**（可选）

格式：
```markdown
# 论文标题

**作者**: ...  
**会议**: ... | **年份**: ...  
**链接**: [arXiv](...)

## 一句话总结
...

## 核心贡献
1. ...
2. ...

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| ... | ... |

## 关键数字

| 指标 | 数值 |
|------|------|
| ... | ... |
```

同时，每个 section 文件**顶部**加返回链接：
```markdown
[← 返回 README](../README.md)
```

---

### 规则 7：添加 Citation Landscape（Semantic Scholar 数据）

每篇论文的 README.md 必须包含 `📊 Citation Landscape` section，数据来源于 Semantic Scholar API（免费，无需 API key）。

#### 包含内容

1. **TLDR**：从 Semantic Scholar 的 `tldr` 字段获取，一句话自动摘要
2. **引用统计**：参考文献数、被引次数、influential citation count
3. **参考文献分组**：按主题分类（Multi-Agent, Memory, RL, Benchmarks 等），每类展示 Top 5，按 citation count 排序
4. **推荐论文**：使用 Recommendations API 获取 10 篇最相关论文

#### API 调用方法

```bash
# 论文详情 + TLDR
curl "https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}?fields=title,year,citationCount,referenceCount,influentialCitationCount,tldr"

# 参考文献（用于分组）
curl "https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}/references?fields=title,year,citationCount,venue,externalIds&limit=1000"

# 推荐论文
curl -X POST "https://api.semanticscholar.org/recommendations/v1/papers/?fields=title,year,citationCount,externalIds,venue" \
  -H "Content-Type: application/json" \
  -d '{"positivePaperIds":["ArXiv:{arxiv_id}"],"negativePaperIds":[]}'
```

#### 注意事项
- 有 rate limit（无 key 约 100 req/5min），调用之间 sleep 2-3s
- Connected Papers 链接：`https://www.connectedpapers.com/main/{arxiv_id}`
- Semantic Scholar 链接：从 paper detail 的 `paperId` 构造

#### 参考示例
见 `/mnt/eason/paper-reading/Agent-Memory/[Arxiv 2602.03036] LatentMem/README.md` 的 📊 Citation Landscape section。

---

### 规则 8：完成后必须推送到 GitHub 并更新大仓库 README

批读完成后：
1. `git add -A` **整个论文目录**（不要只 add sections/，必须包含 images/、full.md、content_list.json、PDF 等所有文件）
2. **更新 `/mnt/eason/paper-reading/README.md`**：确保新论文出现在对应课题的表格中，格式与现有条目一致
3. `git add README.md`
4. `git commit -m "feat: [论文名] 批读完成"`
5. `git push`
6. 把 GitHub 链接发给用户

⚠️ **特别注意**：如果 MinerU 解析和批读分开执行（如主会话解压、子任务批读），确保 images/ 等源文件也被 push。子任务中用 `git add -A 论文目录/` 而非只 `git add sections/`。

---

## 📝 批读格式

**核心特点**：不是"原文"和"理解"分开，而是**边读边批注**。

```markdown
[← 返回 README](../README.md)

# 3. Dataset

## 📌 预览
这个 Section 讲什么，核心要点是...

---

[Section 开头段落原文]

> 💡 **Section 概览**: 这段是整个 Section 的路线图。核心思路是...

---

### 3.1 Data Collection

> 💡 **3.1 要点预览**: 如何从 X 中筛选出 Y？关键是...

[3.1 原文段落]

> 💡 **批注**: 这里的关键点是...

[更多原文]

![Figure 3](../images/xxx.jpg)
*Figure 3: 描述*

> 💡 **Figure 3 批读**:
> - 要点 1
> - 要点 2

> 💡 **3.1 小结**:
> - 要点 1
> - 关键数字: X

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| ... | ... |

### 核心洞察
1. ...
2. ...
```

---

## 🔄 交互式批读工作流

批读是个交互过程，不是一次性生成。

```
Step 1: Agent 生成初版批读（原文 + 基础批注 + README）
         ↓
Step 2: 用户阅读，提问不懂的地方
         ↓
Step 3: Agent 补充详细解释，更新到对应 section 文件
         ↓
       (循环，直到理解透彻)
```

---

## ⚙️ 配置

| 配置项 | 值 |
|--------|-----|
| 论文库位置 | `/mnt/eason/paper-reading/` |
| 目录命名 | `[会议 年份] 论文名` |
| Section 文件命名 | `XX-section-name.md` |
| 批注标记 | `> 💡 **标题**: 内容` |
| GitHub 仓库 | `EasonAI-5589/paper-reading` |

---

## 📂 参考示例

| 论文 | 位置 |
|------|------|
| RoboBrain (CVPR 2025) | `/mnt/eason/paper-reading/VLA/[CVPR 2025] RoboBrain/` |
| RoboBrain 2.0 | `/mnt/eason/paper-reading/VLA/[Arxiv 2507.02029] RoboBrain-2.0/` |

---

## 💡 批读技巧

1. **每段后面加批注**：读完一段就批注，保持思路连贯
2. **subsection 有预览和小结**：预览告诉你要讲什么，小结总结关键点
3. **复杂概念用大白话**：避免术语堆砌，用例子说明
4. **表格画对比图**：ASCII 树形图很直观
5. **交互式补充**：看不懂就问，把回答加到批注里

---

*边读边批，理解更深！📚*
