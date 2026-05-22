from pathlib import Path

from cliver.agents.store import AgentStore


def test_crud(tmp_path: Path):
    store = AgentStore(tmp_path / "test.db")

    # Create
    a = store.create_agent(name="coder", description="Coding agent", model="openai/gpt-4o")
    assert a.name == "coder"

    # List
    agents = store.list_agents()
    assert len(agents) == 1

    # Get by name
    a2 = store.get_agent_by_name("coder")
    assert a2 is not None

    # Update
    updated = store.update_agent(a.id, description="Expert coder")
    assert updated is not None
    assert updated.description == "Expert coder"

    # Set default
    store.create_agent(name="default", is_default=1)
    store.set_default(a.id)
    assert store.get_agent(a.id).is_default == 1
    assert store.get_agent_by_name("default").is_default == 0

    # Delete
    assert store.delete_agent(a.id) is True
    assert len(store.list_agents()) == 1

    store.close()
