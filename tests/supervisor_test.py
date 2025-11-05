import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agent import supervisor, default_config

# Ask the supervisor to generate, analyze, and verify a news piece
out = supervisor.invoke(
    {"messages": [("user", "Create a short article about AI chips, summarize it, analyze sentiment, then verify authenticity.")]},
    config=default_config,
)

# Print full routed conversation
print("\n=== Full history ===")
for m in out["messages"]:
    print(f"[{m.type}] {getattr(m, 'content', '')}")

# Print just the final result
print("\n=== Final result ===")
print(out["messages"][-1].content)