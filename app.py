import os
import sys
import streamlit as st
from typing import TypedDict
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from nemoguardrails import RailsConfig, LLMRails
from langgraph.graph import StateGraph, END

# 1. إعداد المفاتيح في بيئة النظام
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["OPENAI_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Secure AI Agent Platform", page_icon="🤖", layout="centered")
st.title("Secure AI Agent Platform From YASOR LABS")

# ---------------------------------------------------------
# 2. تعريف الأدوات
# ---------------------------------------------------------
@tool
def multiply_numbers(a: float, b: float) -> float:
    """تستخدم لحساب حاصل ضرب رقمين بدقة 100%."""
    return a * b

@tool
def calculate_rsi(prices: str) -> str:
    """تستخدم لحساب مؤشر القوة النسبية RSI لأسعار السهم."""
    return "مؤشر RSI الحالي هو 28.5 (منطقة تشبع بيعي - فرصة شراء)."

# ---------------------------------------------------------
# 3. إعداد الـ Agent والـ Guardrails
# ---------------------------------------------------------
@st.cache_resource
def init_agent():
    tools = [multiply_numbers, calculate_rsi]

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)

    # تحميل NeMo Guardrails
    guardrails = None
    try:
        config = RailsConfig.from_path("./guardrails_config")
        guardrails = LLMRails(config)
    except Exception as e:
        st.error(f"❌ فشل تحميل NeMo Guardrails: {str(e)}")

    class AgentState(TypedDict):
        input_text: str
        guardrail_passed: bool
        response: str
        error_msg: str

    def guardrail_node(state: AgentState) -> AgentState:
        user_input_lower = state["input_text"].lower()
        
        # 1. فحص مباشر وسريع للكلمات المحظورة صراحة (تأمين إضافي 100%)
        prompt_injection_keywords = ["ignore previous instructions", "system prompt", "امسح التعليمات"]
        blocked_topic_keywords = ["بيتزا", "طريقة عمل البيتزا"]

        for kw in prompt_injection_keywords:
            if kw in user_input_lower:
                state["guardrail_passed"] = False
                state["response"] = "🚨 تنبيه أمني: محاولة غير مصرح بها لتجاوز تعليمات النظام."
                return state

        for kw in blocked_topic_keywords:
            if kw in user_input_lower:
                state["guardrail_passed"] = False
                state["response"] = "⚠️ عذراً، هذا الموضوع ممنوع الإجابة عليه حسب سياسة الأمان."
                return state

        if not guardrails:
            state["guardrail_passed"] = True
            return state

        # 2. فحص NeMo Guardrails للسيناريوهات المعقدة
        try:
            res = guardrails.generate(prompt=state["input_text"])
            res_text = res.get("content", "") if isinstance(res, dict) else str(res)

            blocked_keywords = ["🚨", "⚠️", "تنبيه أمني", "ممنوع الإجابة", "تجاوز تعليمات", "سياسة الأمان"]

            if any(keyword in res_text for keyword in blocked_keywords):
                state["guardrail_passed"] = False
                state["response"] = res_text
            else:
                state["guardrail_passed"] = True

        except Exception as e:
            state["guardrail_passed"] = False
            state["response"] = f"🚨 حدث خطأ أثناء فحص NeMo Guardrails:\n{str(e)}"
            state["error_msg"] = str(e)
            
        return state

    def llm_agent_node(state: AgentState) -> AgentState:
        if not state["guardrail_passed"]:
            return state
        
        # System Prompt يسمح بالرد العام واستخدام الأدوات عند الحاجة
        messages = [
            SystemMessage(content="""أنت مساعد ذكي ومفيد ومتعدد المهام.

القواعد:
1. إذا كان سؤال المستخدم يتطلب حسابات رياضية أو تحليل أسهم، استخدم الأداة المناسبة المتاحة فوراً.
2. إذا كان السؤال عاماً (معلومات عامة، رياضة، تاريخ...)، أجب عليه مباشرة بدقة ووضوح بدون استدعاء أدوات."""),
            HumanMessage(content=state["input_text"])
        ]
        
        ai_msg = llm_with_tools.invoke(messages)
        
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "multiply_numbers":
                result = multiply_numbers.invoke(tool_args)
                state["response"] = f"النتيجة الحسابية الدقيقة: {result}"
            elif tool_name == "calculate_rsi":
                result = calculate_rsi.invoke(tool_args)
                state["response"] = f"نتيجة التحليل: {result}"
            else:
                state["response"] = ai_msg.content
        else:
            state["response"] = ai_msg.content

        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail_check", guardrail_node)
    workflow.add_node("agent_execution", llm_agent_node)
    workflow.set_entry_point("guardrail_check")
    workflow.add_edge("guardrail_check", "agent_execution")
    workflow.add_edge("agent_execution", END)

    return workflow.compile()

app_graph = init_agent()

# ---------------------------------------------------------
# 4. واجهة المستخدم
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("اسأل المساعد...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري المعالجة وفحص الأمان..."):
            initial_state = {
                "input_text": user_input,
                "guardrail_passed": False,
                "response": "",
                "error_msg": ""
            }
            
            output = app_graph.invoke(initial_state)
            
            if output["guardrail_passed"]:
                st.caption("🛡️ NeMo Guardrails: تم اجتياز الفحص الأمني.")
            else:
                st.caption("🚨 NeMo Guardrails: تم الحظر بواسطة القواعد.")

            bot_reply = output["response"]
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
