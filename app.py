import os
import asyncio
import streamlit as st
from typing import TypedDict
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from nemoguardrails import RailsConfig, LLMRails
from langgraph.graph import StateGraph, END

 
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Secure AI Agent Platform", page_icon="🤖", layout="centered")

st.title("Secure AI Agent Platform From YASOR LABS")

 
@tool
def multiply_numbers(a: float, b: float) -> float:
   
    return a * b

@tool
def calculate_rsi(prices: str) -> str:
    
    return "مؤشر RSI الحالي هو 28.5 (منطقة تشبع بيعي - فرصة شراء)."

 
@st.cache_resource
def init_agent():
    tools = [multiply_numbers, calculate_rsi]

    # إعداد موديل Groq
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)

    # إعداد NeMo Guardrails وربطه بـ LLM
    config = RailsConfig.from_path("./guardrails_config")
    guardrails = LLMRails(config=config, llm=llm)

    class AgentState(TypedDict):
        input_text: str
        guardrail_passed: bool
        response: str

    async def guardrail_node(state: AgentState) -> AgentState:
        user_input = state["input_text"]
        
        try:
         
            res = await guardrails.generate_async(prompt=user_input)
            
            if isinstance(res, dict):
                res_text = res.get("content", "")
            else:
                res_text = str(res)

            # فحص هل تم إرجاع رسالة حظر من قواعد Colang
            if any(keyword in res_text for keyword in ["تنبيه أمني", "عذراً"]):
                state["guardrail_passed"] = False
                state["response"] = res_text
            else:
                state["guardrail_passed"] = True
        except Exception as e:
            
            state["guardrail_passed"] = False
            state["response"] = f"🚨 خطأ في نظام الأمان: {str(e)}"
            
        return state

    async def llm_agent_node(state: AgentState) -> AgentState:
        if not state["guardrail_passed"]:
            return state
        
        messages = [
            SystemMessage(content="""أنت مساعد ذكي ومفيد ومتعدد المهام.
- إذا كان سؤال المستخدم يتطلب أداة مجهزة (مثل ضرب الأرقام أو حساب مؤشر RSI)، قم باستدعاء الأداة المناسبة.
- إذا كان السؤال عاماً أو استفساراً لا يحتاج إلى أدوات، أجب عليه مباشرة بدقة ووضوح من معرفتك العامة بدون استدعاء أي أداة."""),
            HumanMessage(content=state["input_text"])
        ]
        
        ai_msg = await llm_with_tools.ainvoke(messages)
        
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

        return state
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail_check", guardrail_node)
    workflow.add_node("agent_execution", llm_agent_node)
    workflow.set_entry_point("guardrail_check")
    workflow.add_edge("guardrail_check", "agent_execution")
    workflow.add_edge("agent_execution", END)

    return workflow.compile()

app_graph = init_agent()

 
def run_async_task(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)

 
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("اسأل المساعد (مثال: احسب 15 * 40 أو ما هي عاصمة فرنسا)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري فحص الأمان والتحليل..."):
            initial_state = {"input_text": user_input, "guardrail_passed": False, "response": ""}
            
            output = run_async_task(app_graph.ainvoke(initial_state))
            
            if output["guardrail_passed"]:
                st.caption("NeMo Guardrails: تم اجتياز الفحص الأمني بنجاح.")
            else:
                st.caption("NeMo Guardrails: تم حظر الإدخال بناءً على القواعد الأمنية.")

            bot_reply = output["response"]
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
