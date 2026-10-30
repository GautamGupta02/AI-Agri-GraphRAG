# AI-Agri-GraphRAG
### Graph Retrieval-Augmented Generation for Agricultural Disease Diagnosis Using a Vision-Language Model

AI-Agri-GraphRAG is a multimodal agricultural question-answering system that combines Vision-Language Models (VLMs) with Retrieval-Augmented Generation (RAG) to support crop disease identification, pest recognition, symptom understanding, and agricultural management guidance.

The system integrates image understanding, natural-language queries, and external agricultural knowledge to generate context-aware responses.

## Key Features

- **Multimodal Input:** Accepts crop images with optional text or voice queries.
- **Vision-Language Understanding:** Uses Qwen3-VL-8B-Instruct for image understanding and answer generation.
- **Voice Interaction:** Uses Whisper-based speech transcription.
- **Multiple Retrieval Strategies:** Implements and compares No-RAG, VectorRAG, and GraphRAG.
- **Knowledge-Grounded Responses:** Retrieves relevant agricultural information to support answer generation.
- **Graph-Based Reasoning:** Uses entity retrieval, graph traversal, evidence aggregation, and candidate ranking.

## System Architecture

The project evaluates three configurations:

### 1. No-RAG
The vision-language model generates answers directly from the input image and user query without external knowledge retrieval.

### 2. VectorRAG
Uses dense text embeddings and FAISS similarity search to retrieve relevant agricultural records.

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Vector index: FAISS IndexFlatIP
- Retrieved candidates: Top-5 text records

### 3. GraphRAG
Combines semantic entity retrieval with knowledge-graph traversal to identify related agricultural information and supporting evidence.

- Graph representation: NetworkX MultiDiGraph
- Graph nodes: 136,858
- Graph edges: 75,619
- FAISS entity index: 74,232 vectors
- Entity retrieval: Up to 15 entities
- Associated cases selected: Up to 10

## Technology Stack

| Component | Technology |
|---|---|
| Vision-Language Model | Qwen3-VL-8B-Instruct |
| Speech Transcription | Whisper |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Knowledge Graph | NetworkX |
| Backend / Interface | Python, Gradio |
| Deep Learning Framework | PyTorch |

## Evaluation

The system is evaluated using two agricultural benchmarks:

- **AgMMU-derived evaluation:** 738 matched evaluation cases.
- **MIRAGE:** 502 queries, including 263 identification (MMST-ID) and 239 management (MMST-MG) queries.

### AgMMU Evaluation Results

Normalized exact-match accuracy:

| Method | Correct | Total | Accuracy |
|---|---:|---:|---:|
| No-RAG | 432 | 738 | 58.54% |
| VectorRAG | 483 | 738 | 65.45% |
| GraphRAG | 580 | 738 | **78.59%** |

### MIRAGE Evaluation Results

| Method | MMST-ID | MMST-MG |
|---|---:|---:|
| No-RAG | 0.3023 | 0.6059 |
| VectorRAG | 0.4433 | 0.7473 |
| GraphRAG | **0.5643** | **0.8816** |

GraphRAG achieved the highest aggregate scores among the three evaluated configurations on both evaluation settings.

## Statistical Analysis

To assess the observed performance differences, the study uses:

- Wilson 95% confidence intervals for AgMMU accuracy.
- Exact two-sided McNemar tests for paired binary correctness outcomes.
- Paired Wilcoxon signed-rank tests for MIRAGE scores.
- Paired-bootstrap confidence intervals for MIRAGE mean-score differences.
- Bonferroni correction for multiple pairwise comparisons.

## Retrieval Efficiency

The study also examines retrieval latency and selected retrieval-component storage measurements.

| Metric | VectorRAG | GraphRAG |
|---|---:|---:|
| Mean retrieval latency | 47.690 ms | 31.571 ms |
| Median retrieval latency | 39.601 ms | 25.737 ms |
| 95th percentile | 86.327 ms | 63.773 ms |
| Component loading time | 14.574 s | 26.475 s |
| Selected storage subtotal | 244.07 MiB | 198.48 MiB |

**Note:** These are implementation-specific retrieval measurements, not end-to-end response times. The two retrieval paths use different caching conditions, and the storage figures cover only selected component files.

## Research Objective

The primary objective is to investigate whether explicitly representing relationships among agricultural entities through a knowledge graph provides measurable benefits over conventional semantic vector retrieval in multimodal agricultural question answering.

The study provides an empirical comparison of No-RAG, VectorRAG, and GraphRAG under two complementary evaluation settings.

## Repository Structure

The repository contains the implementation, retrieval components, evaluation scripts, and research materials associated with AI-Agri-GraphRAG.

*The exact directory structure and execution instructions will be documented alongside the corresponding source files.*

## Limitations

- Evaluation is based on the selected benchmarks and experimental configuration.
- Benchmark scores do not independently establish agronomic correctness.
- The retrieval latency comparison uses different caching conditions.
- Further expert validation and component-level ablation studies are needed.

## Research Paper

**AI-Agri-GraphRAG: Graph Retrieval-Augmented Generation for Agricultural Disease Diagnosis Using a Vision-Language Model**

The accompanying research paper describes the system architecture, retrieval methodologies, experimental setup, statistical analysis, and evaluation findings.

