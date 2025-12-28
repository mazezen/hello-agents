
from hello_agents.protocols import A2AServer
import threading
import time

researcher = A2AServer(
        name="researcher",
        description="负责搜索和分析资料的Agent",
        version="1.0.0"
        )


@researcher.skill("research")
def handle_research(text: str) -> str:
    import re
    match = re.search(r'research\s(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text

    result = {
            "topic": topic,
            "findings": f"关于{topic}的研究结果是...",
            "sources": ["来源1", "来源2", "来源3"]
            }
    return str(result)

def start_server():
    researcher.run(host="localhost", port=5000)


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    print("✅ 研究员Agent 服务已启动在http://localhost:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
            print("\n服务已终止")
