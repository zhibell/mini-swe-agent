"""带token预算的 agent： 子类挂 hook， 不修改上游 DefaultAgent"""

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded


class TokenLimitedAgentConfig(AgentConfig):
    """在AgentConfig的基础上加一个 token_limit"""

    token_limit: int = 0  # 0 表示不限制


class TokenLimitedAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=TokenLimitedAgentConfig, **kwargs)
        self.token_used: int = 0

    def query(self) -> dict:
        if self.token_used >= self.config.token_limit > 0:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "TokenLimitExceeded",
                    "extra": {"exit_status": "TokenLimitExceeded", "submission": ""},
                }
            )
        message = super().query()
        self.token_used += message.get("extra", {}).get("response", {}).get("usage", {}).get("total_tokens", 0)
        return message
