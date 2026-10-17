"""Sketch: wrapping a real OpenAI-style client with Cairn.

Not runnable as-is (needs the `openai` package and an API key); it shows the
intended one-line instrumentation pattern for production use.
"""

# pip install openai
from openai import OpenAI  # noqa: F401

from cairn import Session

# client = OpenAI()
#
# def llm(request, model=None, seed=None):
#     """Adapter: Cairn request dict -> OpenAI call -> plain-dict response."""
#     resp = client.chat.completions.create(
#         model=model,
#         messages=request["messages"],
#         seed=seed,                       # recorded in the node!
#         response_format={"type": "json_object"},
#     )
#     msg = resp.choices[0].message
#     return {
#         "content": msg.content,
#         "tool_calls": [
#             {"name": tc.function.name,
#              "arguments": tc.function.arguments,
#              "id": tc.id}
#             for tc in (msg.tool_calls or [])
#         ],
#         "usage": {"prompt": resp.usage.prompt_tokens,
#                   "completion": resp.usage.completion_tokens},
#     }
#
#
# def make_tools():
#     # Map tool names to your real implementations.
#     return {"get_weather": get_weather, "send_email": send_email}
#
#
# with Session.create(".", actor="support-agent") as s:
#     c = s.client(llm=llm, tools=make_tools())
#     c.system("You are a support agent. Be concise.")
#     c.user("Why is my invoice late?")
#     while True:
#         reply = c.infer({"messages": s.messages_as_openai()},  # see note
#                         model="gpt-4o", seed=1234)
#         calls = reply.get("tool_calls") or []
#         if not calls:
#             break
#         for tc in calls:
#             c.tool(tc["name"], json.loads(tc["arguments"]))
#
# NOTE: `Session.messages` gives you the Cairn context list; map it to
# OpenAI's message format in your adapter. Keeping the adapter thin is what
# keeps replay honest: the recorded `request` is exactly what was sent.
