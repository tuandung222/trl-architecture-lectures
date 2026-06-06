---
name: docusaurus-curriculum-builder
description: End-to-end workflow for building, maintaining, and deploying a Docusaurus-based source-code analysis curriculum repository with GitHub Pages deployment.
version: 1.0.0
author: AI Expert & Deep Learning Alignment Specialist
tools_required:
  - node >= 20
  - npm >= 10
  - gh (GitHub CLI) >= 2.0
  - git >= 2.30
  - curl (for verification)
---

# Skill: Docusaurus Curriculum Builder

Kỹ năng này đóng gói toàn bộ quy trình xây dựng một giáo trình Docusaurus chuyên sâu phân tích mã nguồn thư viện (alignment, quantization, inference engine, etc.), từ khâu nghiên cứu source code, viết nội dung học thuật, đến triển khai lên GitHub Pages.

---

## 1. ĐỊNH HƯỚNG TƯ DUY & PHONG CÁCH VIẾT (Persona & Writing Persona)

### 1.1. Persona

Viết với tư cách là **AI Expert, Deep Learning & Alignment Specialist**, đồng thời là một **giảng viên tận tụy**. Mục tiêu là giúp người học nắm bắt gốc rễ toán học, cơ chế hiện thực, và cách sử dụng thực tế của các thuật toán, thay vì chỉ đưa ra tóm tắt bề mặt.

### 1.2. Ngôn ngữ

- **Ngôn ngữ chính**: `{TARGET_LANGUAGE}` (ví dụ: Tiếng Việt).
- **Thuật ngữ kỹ thuật**: Giữ nguyên tiếng Anh khi là chuẩn ngành: *policy, value network, rollout, actor, critic, reference policy, reward model, advantage, importance sampling, KL divergence, loss function, baseline, sequence packing*.
- Giải thích thuật ngữ trước khi sử dụng rộng rãi.
- Tránh ngôn ngữ casual, slang, hoặc jokes. Tông giọng: authoritative, academic, accessible.

### 1.3. Phong cách sư phạm

Với mỗi khái niệm quan trọng, tuân theo luồng sư phạm:

1. **Xung đột cụ thể trước** (concrete tension): Tại sao cần? Vấn đề gì xảy ra nếu không có? (Ví dụ: VRAM OOM, slow generation, reward hacking.)
2. **Toán học**: Xây dựng intuition, công thức (LaTeX), chứng minh kết quả chính.
3. **Pseudocode**: Thuật toán sạch, đọc được.
4. **Mã nguồn thực tế**: Reference file + function name cụ thể trong thư viện mục tiêu.
5. **Hướng dẫn thực hành**: Configuration, code example chạy được.

### 1.4. Quy tắc viết

- **KHÔNG dùng em dash** (—). Dùng comma, colon, semicolon, hoặc parenthesis.
- Dùng `Phần` cho các section khóa học.
- Mermaid diagrams: **KHÔNG** dùng `style`, `fill`, `stroke`, hoặc `color`.
- LaTeX: dùng `$...$` cho inline, `$$...$$` cho block. Giải thích mọi biến số.
- Code blocks: luôn chỉ rõ language (`python`, `typescript`, `bash`).

---

## 2. CÁC RÀNG BUỘC BẢO MẬT & QUYỀN RIÊNG TƯ (Security & Privacy)

### 2.1. README.md

`README.md` phải giữ ở **0 bytes**. Không thêm bất kỳ ký tự nào, kể cả placeholder.

### 2.2. Chặn Search Engine Indexing

**`static/robots.txt`**:
```
User-agent: *
Disallow: /
```

**Docusaurus metadata** (trong `docusaurus.config.ts`):
```typescript
metadata: [
  { name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet' },
]
```

### 2.3. Thông tin nhạy cảm

Public docs **KHÔNG** được chứa:
- Private user instructions hoặc hidden agent constraints.
- Việc `README.md` rỗng.
- Local absolute paths.
- Credentials, tokens, secrets, API keys, hoặc private URLs.
- Tên tài khoản Git hoặc email cụ thể (dùng `{GITHUB_OWNER}`, `{TARGET_EMAIL}` thay thế).

### 2.4. Git Identity

Trước khi commit, luôn set identity **ở repo-local scope**:
```bash
cd {REPOSITORY_PATH}
git config user.name "{TARGET_GIT_USERNAME}"
git config user.email "{TARGET_EMAIL}"
```

---

## 3. QUY TRÌNH THỰC THI & TỰ ĐỘNG HÓA (Execution Workflow)

### 3.1. Phase 1: Khởi tạo Repository

```bash
# Clone reference repo để nghiên cứu cấu trúc
git clone https://github.com/{REFERENCE_OWNER}/{REFERENCE_REPO}.git {REF_DIR}/

# Clone thư viện mục tiêu để phân tích source code
git clone https://github.com/{LIBRARY_OWNER}/{LIBRARY_REPO}.git {LIB_DIR}/

# Tạo GitHub repo mới
gh repo create {GITHUB_OWNER}/{REPOSITORY_NAME} --public --clone
cd {REPOSITORY_NAME}
```

### 3.2. Phase 2: Nghiên cứu Source Code

Đọc các file cốt lõi của thư viện mục tiêu **trước khi viết nội dung**:

| Mức ưu tiên | Loại file | Ví dụ |
|:---|:---|:---|
| 1 | Trainer/Algorithm chính | `{lib}/trainer/grpo_trainer.py` |
| 2 | Config dataclass | `{lib}/trainer/grpo_config.py` |
| 3 | Utility functions | `{lib}/trainer/utils.py` |
| 4 | Data processing | `{lib}/data_utils.py` |
| 5 | Generation/Inference | `{lib}/generation/vllm_generation.py` |
| 6 | Callbacks & monitoring | `{lib}/trainer/callbacks.py` |

### 3.3. Phase 3: Khởi tạo Docusaurus Project

```bash
# Tạo cấu trúc thư mục
mkdir -p docs/ src/pages/ src/css/ static/ .github/workflows/

# Khởi tạo package.json với Docusaurus 3.x
npm init -y
npm install @docusaurus/core@^3 @docusaurus/preset-classic@^3 \
  @docusaurus/theme-mermaid@^3 katex remark-math rehype-katex \
  react react-dom @docusaurus/module-type-aliases @docusaurus/tsconfig \
  @docusaurus/types typescript
```

### 3.4. Phase 4: Cấu hình Docusaurus

**`docusaurus.config.ts`** (các giá trị bắt buộc):
```typescript
const config = {
  title: '{CURRICULUM_TITLE}',
  tagline: '{CURRICULUM_TAGLINE}',
  url: 'https://{GITHUB_OWNER}.github.io',
  baseUrl: '/{REPOSITORY_NAME}/',
  organizationName: '{GITHUB_OWNER}',
  projectName: '{REPOSITORY_NAME}',
  trailingSlash: false,
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  metadata: [
    { name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet' },
  ],
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    ['classic', {
      docs: { sidebarPath: './sidebars.ts' },
      theme: { customCss: './src/css/custom.css' },
    }],
  ],
  markdown: {
    mermaid: true,
  },
};
```

**`sidebars.ts`** (autogenerated):
```typescript
const sidebars = {
  docs: [{ type: 'autogenerated', dirName: '.' }],
};
```

### 3.5. Phase 5: GitHub Actions Deploy

**`.github/workflows/deploy.yml`**:
```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ "{{" }} secrets.GITHUB_TOKEN {{ "}}" }}
          publish_dir: ./build
          user_name: {TARGET_GIT_USERNAME}
          user_email: {TARGET_EMAIL}
```

### 3.6. Phase 6: Viết Nội dung

Cấu trúc giáo trình khuyến nghị:

```
docs/
├── roadmap.md                    (sidebar_position: 1)
├── lesson_0_{topic}.md           (sidebar_position: 2)
├── lesson_1_{topic}.md           (sidebar_position: 3)
├── ...
├── theory_deep_dive/
│   ├── roadmap_theory.md         (sidebar_position: 20)
│   └── theory_N_{topic}.md       (sidebar_position: 21+)
├── case_studies/
│   ├── roadmap_case_studies.md   (sidebar_position: 30)
│   └── case_N_{topic}.md         (sidebar_position: 31+)
└── experiments_deep_dive/
    ├── roadmap_experiments.md    (sidebar_position: 40)
    └── exp_N_{topic}.md          (sidebar_position: 41+)
```

Mỗi bài học tuân theo luồng sư phạm (mục 1.3):
- Problem/tension → Math → Pseudocode → Source code reference → Practical guide

### 3.7. Phase 7: Build & Push

```bash
# Typecheck
npm run typecheck

# Build
npm run build

# Commit và push
git add -A
git commit -m "{COMMIT_MESSAGE}"
git push origin main
```

### 3.8. Phase 8: Enable GitHub Pages

Sau lần push đầu tiên, GitHub Pages cần được **enable thủ công** qua API:

```bash
gh api repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages \
  -X POST \
  -f "build_type=legacy" \
  -f "source[branch]=gh-pages" \
  -f "source[path]=/"
```

> **Quan trọng**: Workflow `peaceiris/actions-gh-pages` push build output sang branch `gh-pages`, nhưng GitHub Pages KHÔNG tự động enable. Phải gọi API trên ít nhất 1 lần.

---

## 4. XỬ LÝ LỖI THƯỜNG GẶP (Troubleshooting Guide)

### 4.1. MDX Compilation Errors

**Lỗi**: `Unexpected character` khi build, thường do angle brackets `<...>` trong markdown table cells.

**Nguyên nhân**: Docusaurus MDX compiler diễn dịch `<...>` như JSX tags.

**Khắc phục**:
- KHÔNG dùng angle brackets trong markdown table cells.
- Thay bằng plain text hoặc escape: `\langle`, `\rangle`.
- Trong code blocks thì OK (được wrap trong triple backticks).

### 4.2. File Write Truncation

**Lỗi**: File bị cắt ngắn khi dùng Write tool, thường tại vị trí chứa `</think>` hoặc `</...>` tags.

**Nguyên nhân**: Tool diễn dịch angle brackets như XML closing tags.

**Khắc phục**:
- Tránh angle brackets trong code comments.
- Nếu cần, dùng SearchReplace thay vì Write cho các file có nội dung phức tạp.
- Rewrite file hoàn toàn nếu bị truncation.

### 4.3. GitHub Pages 404

**Lỗi**: `curl` trả về HTTP 404 mặc dù workflow chạy thành công.

**Nguyên nhân**: GitHub Pages chưa được enable trong repo settings (xem mục 3.8).

**Khắc phục**:
```bash
# Kiểm tra Pages status
gh api repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages

# Nếu 404, enable nó
gh api repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages \
  -X POST -f "build_type=legacy" \
  -f "source[branch]=gh-pages" -f "source[path]=/"

# Đợi 30-60 giây rồi kiểm tra
sleep 30 && curl -s -o /dev/null -w "%{http_code}" \
  https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/
```

### 4.4. Broken Cross-Reference Links

**Lỗi**: Links giữa các bài học không hoạt động.

**Nguyên nhân**: Relative path sai. Cấu trúc thư mục Docusaurus:
- Từ `docs/lesson_X.md` đến `docs/theory_deep_dive/theory_Y.md`: dùng `./theory_deep_dive/theory_Y`
- Từ `docs/theory_deep_dive/theory_Y.md` đến `docs/lesson_X.md`: dùng `../lesson_X`
- Docusaurus config `onBrokenLinks: 'warn'` chỉ cảnh báo, không block build.

**Khắc phục**:
- Luôn test build sau khi thêm cross-references.
- Dùng relative paths (không dùng absolute paths).

### 4.5. Sidebar Ordering Issues

**Lỗi**: Bài học mới không xuất hiện đúng vị trí trong sidebar.

**Khắc phục**:
- Dùng `sidebar_position` trong YAML frontmatter (numeric sort).
- Có thể dùng fractional values (ví dụ: `6.5`) để chèn giữa 2 bài hiện có mà không cần renumber.
- Nếu dùng autogenerated sidebars, file mới trong existing directories sẽ tự động xuất hiện.

### 4.6. `npm ci` Fails in CI

**Lỗi**: `npm ci` fail vì `package-lock.json` không match `package.json`.

**Khắc phục**:
- Luôn commit cả `package-lock.json` sau khi `npm install`.
- Hoặc dùng `npm install` thay cho `npm ci` trong workflow (chậm hơn nhưng forgiving hơn).

---

## 5. TIÊU CHUẨN XÁC MINH HOÀN THÀNH (Verification Checklist)

Trước khi bàn giao, chạy toàn bộ các kiểm tra sau:

### 5.1. Build Integrity

```bash
# Typecheck (nếu có TypeScript)
npm run typecheck

# Build static site
npm run build
# Expected: "Generated static files in build" with exit code 0
```

### 5.2. Content Quality

```bash
# Kiểm tra không có em dash
grep -r '—' docs/ src/ --include="*.md" --include="*.tsx"
# Expected: no matches

# Kiểm tra README.md rỗng
wc -c README.md
# Expected: 0

# Kiểm tra robots.txt
cat static/robots.txt
# Expected: "User-agent: *\nDisallow: /"

# Line count overview
wc -l docs/*.md docs/**/*.md | sort -n
# Review: no lesson should be below minimum threshold
```

### 5.3. Deployment Verification

```bash
# Check GitHub Actions workflow status
gh run list --limit 5
# Expected: latest run shows "completed success"

# Check GitHub Pages is enabled
gh api repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages
# Expected: returns JSON with "status" and "html_url"

# Check live site HTTP status
curl -s -o /dev/null -w "%{http_code}" \
  https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/
# Expected: 200

# Check noindex meta tag
curl -s https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/ | grep -o 'noindex'
# Expected: "noindex"
```

### 5.4. Git & Push Verification

```bash
# Check working tree clean
git status --short
# Expected: empty output

# Check remote is set correctly
git remote -v
# Expected: origin points to github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}

# Check latest commit on remote
git log --oneline -1 origin/main
# Expected: matches latest local commit
```

### 5.5. Cross-Reference Verification

```bash
# Extract all internal links from markdown files
grep -roh '\[.*\](\./[^)]*)' docs/ | sort | uniq
# Review: manually verify 5-10 random links resolve correctly

# Check for broken link warnings in build output
npm run build 2>&1 | grep -i "broken"
# Expected: no output
```

---

## Appendix A: Template YAML Frontmatter cho Lesson Files

```yaml
---
sidebar_position: {N}
sidebar_label: "Bài {N}: {SHORT_LABEL}"
---
```

## Appendix B: Cross-Reference Section Template

Thêm vào cuối mỗi bài học (trước dòng "Bài tiếp theo..."):

```markdown
---

## Xem thêm

- [Lý thuyết X: {Topic}](./theory_deep_dive/theory_X_{topic}.md): {Mô tả ngắn}
- [Case Study: {Topic}](./case_studies/case_X_{topic}.md): {Mô tả ngắn}
- [Experiment: {Topic}](./experiments_deep_dive/exp_X_{topic}.md): {Mô tả ngắn}

Bài tiếp theo phân tích {next_lesson_topic}.
```

## Appendix C: Environment Variables Reference

| Variable | Description | Example |
|:---|:---|:---|
| `{GITHUB_OWNER}` | GitHub username/org | `tuandung222` |
| `{REPOSITORY_NAME}` | Target repo name | `trl-architecture-lectures` |
| `{TARGET_GIT_USERNAME}` | Git display name | `tuandung222` |
| `{TARGET_EMAIL}` | Git email | `{id}+{user}@users.noreply.github.com` |
| `{CURRICULUM_TITLE}` | Site title | `TRL Internals` |
| `{CURRICULUM_TAGLINE}` | Site tagline | Vietnamese description |
| `{TARGET_LANGUAGE}` | Primary content language | `Vietnamese` |
| `{LIBRARY_OWNER}` | Target library GitHub owner | `huggingface` |
| `{LIBRARY_REPO}` | Target library repo name | `trl` |
| `{REFERENCE_OWNER}` | Reference curriculum owner | `tuandung222` |
| `{REFERENCE_REPO}` | Reference curriculum repo | `verl-architecture-lectures` |
| `{REPOSITORY_PATH}` | Local clone path | `/path/to/repos/{REPOSITORY_NAME}` |
| `{COMMIT_MESSAGE}` | Descriptive commit msg | `feat: add RLOO lesson` |
