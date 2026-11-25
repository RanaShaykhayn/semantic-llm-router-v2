# Semantic LLM Router with ReAct Agent

A cost-aware routing system I built to optimize LLM API usage by directing queries to the most appropriate model.

## 🎯 The Problem I Solved
Sending every user query to a large, expensive model (like Gemini Pro) wastes money and time. Simple queries (like "hello") don't need complex reasoning.

## 💡 My Solution
1. **Semantic Router**: I embedded queries and compared them using Cosine Similarity to route them to either Flash (cheap/fast) or Pro (smart).
2. **Guardrails**: I added a security layer to block Prompt Injection attacks.
3. **ReAct Agent**: For complex queries, I built an agent that reasons, uses the router as a tool, and observes the result before answering.
4. **Hybrid Router**: I created a hybrid system that uses fast routing for simple questions and switches to the Agent for complex ones.
5. **Docker**: I containerized the entire application so it can run anywhere.

## 📊 What I Achieved
- **100% Accuracy**: I tested 10 queries (English & Arabic) and achieved 100% routing accuracy.
- **Agent Loop**: The agent successfully uses tools and stops when it has enough information.
- **Performance Analysis**: I discovered that Embedding takes 41% of total latency, not generation.

## 🌐 Live Demo
You can try the live application here: [Semantic LLM Router](https://semantic-llm-router-v2-kz6cjjbfdrdf4juir77tpp.streamlit.app/)

-

## 🚀 How to Run It