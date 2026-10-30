"""The project list, folder browsing and opening a project in place."""

import asyncio
import hashlib
from pathlib import Path

import pytest
from instinct.harness.projects import Projects, browse, state_folder
from instinct.harness.service import AppService


def test_projects_are_recorded_renamed_pinned_and_removed(tmp_path):
    projects = Projects(tmp_path / "projects.json")
    first, second = tmp_path / "alpha", tmp_path / "beta"
    first.mkdir()
    second.mkdir()
    a = projects.touch(first)
    b = projects.touch(second)
    assert a["title"] == "alpha" and projects.touch(first)["id"] == a["id"]
    projects.update(b["id"], title="  Beta   app ")
    projects.update(b["id"], pinned=True)
    listed = Projects(tmp_path / "projects.json").list()
    assert [p["title"] for p in listed] == ["Beta app", "alpha"]  # pinned first
    with pytest.raises(ValueError):
        projects.update(a["id"], title=" ")
    second.rmdir()
    assert projects.list()[0]["missing"]
    projects.remove(b["id"])
    assert [p["id"] for p in projects.list()] == [a["id"]]
    with pytest.raises(ValueError):
        projects.touch(tmp_path / "nowhere")


def test_state_folders_match_the_mac_app(tmp_path):
    folder = tmp_path / "code"
    folder.mkdir()
    digest = hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    assert (
        state_folder(tmp_path / "home", folder)
        == tmp_path / "home" / "workspaces" / f"code-{digest}"
    )


def test_browsing_stays_inside_home(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")
    listing = browse(None, root=tmp_path)
    assert [f["name"] for f in listing["folders"]] == ["src"] and listing[
        "parent"
    ] is None
    assert browse(str(tmp_path / "src"), root=tmp_path)["parent"] == str(tmp_path)
    with pytest.raises(ValueError):
        browse("/", root=tmp_path)


def test_opening_a_project_switches_in_place(tmp_path):
    home = tmp_path / "home"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    projects = Projects(home / "projects.json")

    async def run():
        current = {}

        def build(folder: Path) -> AppService:
            service = AppService(
                folder, state_folder(home, folder), demo=True, projects=projects
            )
            service.switch_project = switch
            return service

        async def switch(folder: Path) -> None:
            previous = current["service"]
            current["service"] = build(folder)
            current["service"].adopt(previous)
            await previous.close()

        current["service"] = build(first)
        service = current["service"]
        listed = await service.dispatch("GET", "/api/projects", {})
        assert listed["current"] == listed["projects"][0]["id"]
        opened = await service.dispatch(
            "POST", "/api/projects", {"path": str(second), "open": True}
        )
        assert opened["switched"] and current["service"].workspace == second.resolve()
        assert service._closed
        listed = await current["service"].dispatch("GET", "/api/projects", {})
        assert listed["current"] == opened["project"]["id"]
        with pytest.raises(ValueError, match="open project"):
            await current["service"].dispatch(
                "POST", f"/api/projects/{listed['current']}/delete", {}
            )
        folders = await current["service"].dispatch(
            "GET", "/api/folders", {"path": str(Path.home())}
        )
        assert folders["path"] == str(Path.home())
        await current["service"].close()

    asyncio.run(run())
