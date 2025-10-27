import pytest


@pytest.fixture()
def workflow_path(tmp_path):
    workflow_path = tmp_path / "workflows"
    workflow_path.mkdir()
    return workflow_path


@pytest.fixture()
def workflow_cache_path(tmp_path):
    workflow_cache_path = tmp_path / "workflows_exec_cache"
    workflow_cache_path.mkdir()
    return workflow_cache_path
