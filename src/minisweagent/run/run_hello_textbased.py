"""用文本模式模型跑 hello_world：修复 No tool calls found 问题。"""

import logging

import typer
import yaml

from minisweagent import package_dir
from minisweagent.agents.token_limited import TokenLimitedAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


def print_usage(messages: list[dict]) -> None:
    """任务 1.3 后半·取证：打印每次模型调用的 token 账单。"""
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        extra = msg.get("extra", {})
        print(f"[{i}] extra 第一层的键: {list(extra.keys())}")
        usage = extra.get("response", {}).get("usage", {})
        print(f"[{i}] usage: {usage}")


def print_messages(messages: list[dict], max_chars: int = 80) -> None:
    """任务 1.2 插桩：打印每条消息的角色与内容摘要。

    参数:
        messages: agent.messages，list[dict]，每条含 role/content，可能有 extra
        max_chars: content 超过该长度则截断，防止刷屏
    """
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        print(f"[{i}] {role}: {content}")
        if role == "exit":
            exit_status = msg.get("extra", {}).get("exit_status", "")
            print(f"\nExit status:{exit_status}")


app = typer.Typer()


@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", prompt=True),
    model_name: str = typer.Option("deepseek/deepseek-flash", "-m", "--model"),
    step_limit: int | None = typer.Option(None, "--step-limit", help="步数上限，0 = 不限制"),
    token_limit: int | None = typer.Option(None, "--token-limit", help="token 上限， 0 = 不限制"),
):
    logging.basicConfig(level=logging.DEBUG)
    agent_cfg: dict = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())["agent"]
    if step_limit is not None:
        agent_cfg = {**agent_cfg, "step_limit": step_limit}
    if token_limit is not None:
        agent_cfg = {**agent_cfg, "token_limit": token_limit}
    agent = TokenLimitedAgent(
        LitellmTextbasedModel(model_name=model_name),
        LocalEnvironment(),
        **agent_cfg,
    )
    agent.run(task)
    print_messages(agent.messages, max_chars=80)
    print_usage(agent.messages)


if __name__ == "__main__":
    app()
