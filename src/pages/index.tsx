import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero custom-hero', styles.heroBanner)}>
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'inline-flex', padding: '4px 12px', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', borderRadius: '100px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(16, 185, 129, 0.25)', marginBottom: '1.5rem' }}>
          DOCUMENTATION & PRACTICAL ALIGNMENT COURSE
        </div>
        <Heading as="h1" className="hero__title" style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #fff 30%, #34d399 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '1.5rem' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ maxWidth: '750px', margin: '0 auto 2.5rem auto', fontSize: '1.2rem', lineHeight: '1.6', opacity: 0.85 }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)', transition: 'all 0.3s ease' }}
            to="/docs/roadmap">
            Bắt đầu học ngay 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', marginLeft: '1rem', border: '1px solid var(--card-border)' }}
            href="https://github.com/tuandung222/trl-architecture-lectures">
            View on GitHub 🛠️
          </Link>
        </div>
      </div>
    </header>
  );
}

interface FeatureItem {
  title: string;
  badge: string;
  description: string;
}

const CorePillars: FeatureItem[] = [
  {
    title: 'HuggingFace Ecosystem',
    badge: 'Design Philosophy',
    description: 'Tích hợp sâu với Transformers, Accelerate, PEFT và Datasets, cho phép tận dụng toàn bộ hạ tầng Hugging Face cho quy trình alignment từ SFT đến RLHF.',
  },
  {
    title: 'Modular Trainer Architecture',
    badge: 'Code Architecture',
    description: 'Kiến trúc Trainer kế thừa từ HuggingFace Trainer với _BaseTrainer, hệ thống Config dataclass, Callbacks và DataCollator tùy biến cho từng thuật toán alignment.',
  },
  {
    title: 'vLLM & Multi-turn Integration',
    badge: 'Production Ready',
    description: 'Tích hợp vLLM cho generation tốc độ cao, hỗ trợ tool-calling đa lượt (multi-turn), environment factory và importance sampling correction cho off-policy training.',
  },
];

interface LectureItem {
  number: string;
  title: string;
  desc: string;
  path: string;
  category: 'Background' | 'Core Theory' | 'Deep Dive Code' | 'Optimization' | 'Practice';
}

const Lectures: LectureItem[] = [
  {
    number: 'Bài 0',
    title: 'Nền tảng Alignment & Toàn cảnh RLHF',
    desc: 'Tại sao LLM cần Alignment? So sánh SFT, DPO, RLHF. Tổng quan hệ sinh thái Hugging Face và vị trí của TRL trong pipeline huấn luyện LLM.',
    path: '/docs/lesson_0_alignment_fundamentals',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'Kiến trúc TRL & Triết lý Thiết kế',
    desc: 'Phân tích cấu trúc module, lazy import, _BaseTrainer kế thừa từ HF Trainer. Hệ thống Config dataclass và cách TRL mở rộng Transformers.',
    path: '/docs/lesson_1_trl_architecture',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'SFT Trainer - Supervised Fine-Tuning Deep Dive',
    desc: 'DataCollator cho SFT, kỹ thuật sequence packing, chat template processing. Phân tích luồng dữ liệu từ dataset đến gradient update.',
    path: '/docs/lesson_2_sft_trainer',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: 'DPO & Preference Optimization - Từ Toán đến Code',
    desc: 'Toán học DPO chi tiết, các biến thể (CPO, ORPO, IPO, KTO). Reference model management, DataCollator cho preference data.',
    path: '/docs/lesson_3_dpo_preference',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'GRPO Internals - Group Relative Policy Optimization',
    desc: 'Phân tích chi tiết GRPOTrainer: rollout generation, reward computation, advantage calculation, loss variants (GRPO, DAPO, CISPO, VESPO, SAPO).',
    path: '/docs/lesson_4_grpo_internals',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 5',
    title: 'PPO & Online RL - Experimental Trainers',
    desc: 'Kiến trúc PPOTrainer trong trl/experimental: value head, GAE computation, reward model integration, rollout buffer management.',
    path: '/docs/lesson_5_ppo_online_rl',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6',
    title: 'Reward Engineering & Verification',
    desc: 'Hệ thống reward functions: accuracy_reward (math_verify), format_reward, custom reward. Async reward, multi-objective aggregation.',
    path: '/docs/lesson_6_reward_engineering',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 7',
    title: 'vLLM Integration & Generation Optimization',
    desc: 'VLLMGeneration class, colocation mode, importance sampling correction, off-policy sequence masking. Tối ưu generation throughput.',
    path: '/docs/lesson_7_vllm_generation',
    category: 'Optimization'
  },
  {
    number: 'Bài 8',
    title: 'Thực hành: Tự viết Toy Alignment Pipeline',
    desc: 'Lập trình từ đầu một pipeline DPO/GRPO tối giản bằng PyTorch thuần, mô phỏng các thành phần cốt lõi của TRL trainers.',
    path: '/docs/lesson_8_toy_alignment_pipeline',
    category: 'Practice'
  }
];

function CategoryBadge({ category }: { category: LectureItem['category'] }) {
  const colors: Record<LectureItem['category'], { bg: string, text: string }> = {
    'Background': { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa' },
    'Core Theory': { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399' },
    'Deep Dive Code': { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24' },
    'Optimization': { bg: 'rgba(236, 72, 153, 0.15)', text: '#f472b6' },
    'Practice': { bg: 'rgba(139, 92, 246, 0.15)', text: '#a78bfa' },
  };

  const color = colors[category];

  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: color.bg, color: color.text, alignSelf: 'flex-start' }}>
      {category}
    </span>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} | Deep Dive TRL Alignment Architecture`}
      description="Chuỗi bài giảng phân tích chi tiết kiến trúc, thuật toán và mã nguồn của thư viện TRL (Transformer Reinforcement Learning) dành cho AI Alignment & Systems Engineers.">
      <HomepageHeader />
      
      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              🚀 Ba Trụ Cột Kiến Trúc Của TRL
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Những cột trụ thiết kế giúp TRL trở thành thư viện alignment phổ biến nhất trong hệ sinh thái Hugging Face
            </p>
          </div>
          
          <div className="row">
            {CorePillars.map((item, idx) => (
              <div key={idx} className="col col--4" style={{ marginBottom: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--ifm-color-primary)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>
                    {item.badge}
                  </span>
                  <Heading as="h3" style={{ fontSize: '1.35rem', marginBottom: '1rem', fontWeight: 600 }}>
                    {item.title}
                  </Heading>
                  <p style={{ opacity: 0.8, fontSize: '0.95rem', lineHeight: '1.6', margin: 0 }}>
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Lectures Curriculum Section */}
        <section className="container">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              📚 Giáo Trình Học Tập (Curriculum)
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Đi từ kiến thức nền tảng alignment, phân tích sâu cấu trúc Trainer, mã nguồn các thuật toán đến thực hành viết pipeline.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {Lectures.map((lecture, idx) => (
              <Link 
                to={lecture.path} 
                key={idx} 
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <div className="glass-panel" style={{ padding: '1.5rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', cursor: 'pointer' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--ifm-color-primary)' }}>
                        {lecture.number}
                      </span>
                      <CategoryBadge category={lecture.category} />
                    </div>
                    <Heading as="h3" style={{ fontSize: '1.15rem', marginBottom: '0.75rem', fontWeight: 600, lineHeight: '1.4' }}>
                      {lecture.title}
                    </Heading>
                    <p style={{ opacity: 0.8, fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                      {lecture.desc}
                    </p>
                  </div>
                  <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', fontSize: '0.85rem', fontWeight: 600, color: 'var(--ifm-color-primary-light)' }}>
                    Đọc bài học này <span style={{ marginLeft: '4px', transition: 'transform 0.2s ease' }} className="arrow-icon">→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Practice Engine Banner */}
        <section className="container" style={{ marginTop: '6rem' }}>
          <div className="glass-panel" style={{ padding: '3rem', background: 'radial-gradient(circle at 90% 10%, rgba(16, 185, 129, 0.12) 0%, transparent 60%), var(--card-bg)', textAlign: 'center', borderRadius: '16px' }}>
            <Heading as="h2" style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '1rem' }}>
              💻 Thực Hành: Tự Xây Dựng Toy Alignment Pipeline
            </Heading>
            <p style={{ maxWidth: '700px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Trong Bài 8, chúng ta sẽ tự tay triển khai bằng PyTorch thuần một pipeline DPO và GRPO tối giản, mô phỏng chính xác các thành phần cốt lõi của TRL trainers từ con số 0.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_8_toy_alignment_pipeline">
                Đến Bài Học Thực Hành
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
