"""用文本模式模型跑 hello_world：修复 No tool calls found 问题。"""
import yaml
from pathlib import Path
import typer
import logging

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel   # ← 空 1：补全模块名和类名（见诊断第三节）

app = typer.Typer()

@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", prompt=True),
    model_name: str = typer.Option("deepseek/deepseek-flash", "-m", "--model"),
):
    logging.basicConfig(level=logging.DEBUG)
    agent = DefaultAgent(
        LitellmTextbasedModel(model_name=model_name),   # ← 空 2：把模型构造换成文本模式类
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(task) # ← 空 3：把打印消息的函数传给 run 方法
    print_messages(agent.messages, max_chars=80)  # ← 空 4：打印消息

def print_messages(messages:list[dict],max_chars: int = 80) -> None:
    """任务 1.2 插桩：打印每条消息的角色与内容摘要。

    参数:
        messages: agent.messages，list[dict]，每条含 role/content，可能有 extra
        max_chars: content 超过该长度则截断，防止刷屏
    """
    for i,msg in enumerate(messages):
        role = msg.get("role","unknown")
        content = msg.get("content","")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
            print(f"[{i}] {role}: {content}")
        if role == "exit":
            exit_status = msg.get("extra","").get("exit_status","")
            print(f"\nExit status:{exit_status}")


if __name__ == "__main__":
    app()

