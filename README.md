# 🛡️ Secure AI Agent Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![NeMo Guardrails](https://img.shields.io/badge/Security-NVIDIA%20NeMo-green.svg)](https://github.com/NVIDIA/NeMo-Guardrails)
[![Groq](https://img.shields.io/badge/Inference-Groq%20Llama%203.3-purple.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A production-ready, security-first Agentic AI Architecture engineered to eliminate prompt injection vulnerabilities and constrain LLM behaviors within domain-specific boundaries using NVIDIA NeMo Guardrails and LangGraph state machines. Powered by ultra-low-latency inference via Groq (Llama 3.3 70B).

---

## 📐 System Architecture

`mermaid
graph TD
    User([👤 User Request]) --> GuardrailNode[🛡️ NeMo Guardrails Node]
    
    GuardrailNode -->|Prompt Injection / Off-Topic| Blocked[🚨 Security / Topic Alert Response]
    GuardrailNode -->|Passed Security Policy| AgentNode[🧠 LangGraph Agent Node]
    
    AgentNode -->|Requires Computation| ToolCall{🛠️ Tool Selector}
    ToolCall -->|Math Query| CalcTool[🔢 Math Calculator Tool]
    ToolCall -->|Technical Analysis| RSITool[📈 Stock RSI Analyzer Tool]
    
    CalcTool --> FormatResponse[📝 Response Formatter]
    RSITool --> FormatResponse
    AgentNode -->|Direct Answer| FormatResponse
    
    FormatResponse --> FinalOutput([💬 Secure Output to User])
    Blocked --> FinalOutput
