"""用文本模式模型跑 hello_world：修复 No tool calls found 问题。"""
import yaml
from pathlib import Path
import typer

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
    agent = DefaultAgent(
        LitellmTextbasedModel(model_name=model_name),   # ← 空 2：把模型构造换成文本模式类
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(task)

if __name__ == "__main__":
    app()
