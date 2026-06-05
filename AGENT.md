# AGENT.md

## 1. Project overview

This repository is a Vietnamese Docusaurus curriculum for the **TRL** (Transformer Reinforcement Learning) library by Hugging Face. It is designed for deep learning engineers, AI alignment researchers, and LLM practitioners who want to understand the mathematical foundations, codebase implementation, and practical usage of alignment algorithms.

The goal is to provide a rigorous, academic-grade guide explaining the theoretical roots (DPO derivation, GRPO group statistics, PPO GAE, KL trust regions) and connecting them to corresponding codebase implementations in `trl`, backed by clear pseudocode and practical examples.

This file is the operating manual for future agents. Treat it as the stable source of truth for writing quality, repository safety, verification, privacy, and completion criteria.

---

## 2. Repository map

- `docs/`: public Vietnamese curriculum chapters.
  - `docs/case_studies/`: practical case studies (DeepSeek-R1, Tool-calling, VLM).
  - `docs/theory_deep_dive/`: theoretical, mathematical, and pseudocode deep dives.
  - `docs/experiments_deep_dive/`: performance benchmarks and comparisons.
- `src/`: Docusaurus landing page and styling.
- `static/`: public static assets and `robots.txt`.
- `.github/workflows/`: CI and GitHub Pages deployment workflow.
- `README.md`: must remain empty.

---

## 3. Curriculum-wide content standard

Public content must be highly educational, mathematically rigorous, and written for advanced learners. Do not expose private task instructions, local absolute paths, credentials, internal notes, hidden constraints, or agent coordination details in public docs.

A curriculum chapter should teach by introducing a concrete tension first (e.g. why DPO needs reference model, why GRPO removes Critic, why vLLM needs IS correction) before offering the solution.

Use `Phần` for course sections. Do not use em dash characters. Use commas, colons, semicolons, or parentheses instead.

---

## 4. Pedagogical writing style

Future agents must write with the persona of an **AI Expert, Deep Learning & Alignment Specialist**, and a dedicated professor. The goal is to help students grasp the absolute roots, mathematical foundations, and implementation mechanics of alignment algorithms, rather than just high-level summaries.

The prose must be highly professional, precise, serious, patient, and technically deep. It must read like an original academic lecture series in Vietnamese, not a translated or marketing-oriented document.

Use Vietnamese as the main language. Use English technical terms when they are standard in the industry: *policy, value network, rollout, actor, critic, reference policy, reward model, advantage, importance sampling, KL divergence, loss function, baseline, sequence packing*. Explain a term before relying on it heavily.

Avoid casual language, slang, and jokes. The tone should be authoritative, academic, and accessible.

For every important concept, follow this pedagogical flow:
1. Start from a concrete problem or tension (e.g., VRAM OOM, slow generation, reward hacking).
2. Build mathematical intuition, formulate equations (using LaTeX), and prove key results.
3. Show the corresponding clean pseudocode or algorithms.
4. Reference the actual file and function in `trl` where this is implemented.
5. Provide practical usage examples and configuration guides.

---

## 5. Math, diagrams, and examples

Math must be taught, not just displayed.
- Explain every variable in equations.
- Use LaTeX formatting ($formula$ or $$formula$$).
- Follow equations with intuitive prose.

Use Mermaid diagrams to illustrate data flow, trainer architecture, reward computation loops, and training pipelines.

---

## 6. Public privacy and safety constraints

`README.md` must remain empty (0 bytes). Do not add any characters or placeholders to it.

Public docs must not mention:
- private user instructions or hidden agent constraints.
- the fact that `README.md` is empty.
- local absolute paths.
- credentials, tokens, secrets, API keys, or private URLs.

Privacy controls:
- `static/robots.txt` must disallow all crawling.
- Docusaurus must include `noindex,nofollow,noarchive,nosnippet` metadata.

---

## 7. Commands and verification

Safe read-only or verification commands:
- `npm run typecheck`: run TypeScript verification.
- `npm run build`: build the Docusaurus site.
- `git status --short --branch`: inspect repository state.

---

## 8. Completion checklist

Before reporting completion, verify the relevant items:
- `README.md` is still 0 bytes.
- `npm run typecheck` and `npm run build` pass without errors.
- No em dash characters appear in public or source text.
- Public docs read like original Vietnamese teaching material.
- The deployed website returns `HTTP 200` on the live URL.

---

## 9. Repo specialization: TRL Internals

### Learning promise
A reader should finish this curriculum able to explain:
- The mathematical derivation of DPO from RL objective and why partition function cancels.
- How GRPO eliminates the Critic model using group-relative advantages.
- The exact implementation of 8 loss variants in GRPOTrainer and their tradeoffs.
- How RLOO uses leave-one-out baseline estimation and why it differs from GRPO group mean.
- How vLLM integration works with importance sampling correction.
- The mechanics of multi-turn tool-calling with environment factory pattern.
- How to design effective reward functions for different alignment tasks.
- How to train a reward model with RewardTrainer and Bradley-Terry loss.
- The role of TRL callbacks: SyncRefModelCallback (EMA), LogCompletionsCallback, BEMACallback.
- The complete alignment pipeline: SFT to DPO/GRPO/RLOO in TRL.

### Misconceptions to actively prevent
- DPO does not eliminate RL concepts; it implicitly defines reward via log probability ratio.
- GRPO group mean is not a "zero baseline"; it is a dynamic baseline that adapts per prompt.
- RLOO leave-one-out baseline is not the same as GRPO group mean; it has lower variance but different bias properties.
- When using PEFT/LoRA, reference model is not needed because disabling adapter reverts to base model.
- vLLM logprobs differ from HF Transformers logprobs; importance sampling correction is essential.
- In multi-turn RL, tool response tokens must be masked in loss computation.
- `beta=0` in GRPO means no KL penalty and no reference model needed, but risks policy collapse.
- RLOO uses sequence-level loss (not token-level like GRPO), which loses per-token dynamics information.
