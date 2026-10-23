"""CrewAI integration for Cairn.

Provides callbacks to record CrewAI agent steps and task executions::

    from crewai import Agent, Crew, Task
    from cairn import Session
    from cairn.integrations.crewai import CairnCrewCallback

    session = Session.create("./agent-repo", actor="crewai-leader")
    cb = CairnCrewCallback(session)

    agent = Agent(
        role="Researcher",
        goal="Discover breakthroughs",
        step_callback=cb.on_step,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        task_callback=cb.on_task,
    )
"""

from __future__ import annotations

from typing import Any, Optional

from ..session import Session


class CairnCrewCallback:
    """Callback hook for CrewAI agents and tasks."""

    def __init__(self, session: Session):
        self.session = session

    def on_step(self, step_output: Any) -> None:
        """Invoked after each CrewAI agent step."""
        text = str(step_output)
        agent_name = getattr(step_output, "agent", "crew-agent")
        self.session.observation(
            source=f"crewai:{agent_name}",
            data={"step": text},
        )

    def on_task(self, task_output: Any) -> None:
        """Invoked after each CrewAI task completes."""
        description = getattr(task_output, "description", "task")
        raw_output = getattr(task_output, "raw", str(task_output))
        self.session.tool_result(
            name=f"task:{description[:40]}",
            result={"output": raw_output},
            ok=True,
        )
