
from re import IGNORECASE
from hello_agents.protocols import A2AServer, A2AClient

import threading
import time

researcher = A2AServer(
        name="researcher",
        description="研究员"
        )

@researcher.skill("research")
def do_research(text: str) -> str:
    import re
    match = re.search(r"research\s+(.+)", text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    return str({"topic": topic, "findings": f"{topic}的研究成果"})

writer = A2AServer(
        name='writer',
        description="撰写员"
        )


@writer.skill("write")
def wirite_article(text: str) -> str:
    import re
    match = re.search(r'write\s+(.+)', text, re.IGNORECASE)
    content = match.group(1).strip() if match else text

    try:
        data = eval(content)
        topic = data.get("topic", "未知主题")
        findings = data.get("findings", "无研究成果")
    except:
        topic = "未知主题"
        findings = content

    return f"# {topic}\n\n基于研究: {findings}\n\n文章内容..."


editor = A2AServer(
        name="editor",
        description="编辑"
        )


@editor.skill("edit")
def edit_article(text: str) -> str:
    import re
    match = re.search(r'edit\s+(.+)', text, re.IGNORECASE)
    article = match.group(1).strip() if match else text

    result = {
            "article": article + '\n\n[已编辑优化]',
            'feedback': "文章质量良好",
            "approved": True
            }

    return str(result)


threading.Thread(target=lambda: researcher.run(port=5000), daemon=True).start()
threading.Thread(target=lambda: writer.run(port=5001), daemon=True).start()
threading.Thread(target=lambda: editor.run(port=5002), daemon=True).start()
time.sleep(2)

research_client = A2AClient("http://localhost:5000")
writer_client = A2AClient("http://localhost:5001")
editor_client = A2AClient("http://localhost:5002")


def create_content(topic):
    research = research_client.execute_skill("research", f"research {topic}")
    research_Data = research.get('result', '')

    article = writer_client.execute_skill('write', f"write {research_Data}")
    article_content = article.get('result', '')

    final = editor_client.execute_skill('edit', f"edit {article_content}")
    return final.get('result', '')


result = create_content('AI在医疗领域的应用')
print(f"\n最终结果: \n {result}")

