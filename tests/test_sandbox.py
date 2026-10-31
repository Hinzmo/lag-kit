"""Sandbox tests.

Two layers, tested differently. Path checks and the deny-list are pure logic and run
everywhere. The container is the actual boundary, so there is one test that starts a
real one and tries to break out — a mocked escape test proves nothing about whether
the escape works.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lag.sandbox import PROJECT_MNT, Sandbox, SandboxedProvider, available


class FakeInner:
    def __init__(self):
        self.called = []

    def schemas(self):
        return ["s"]

    def call(self, name, args):
        self.called.append((name, args))
        return f"inner ran {name}"


@pytest.fixture
def sb(tmp_path):
    return Sandbox(tmp_path / "ws", read_root=tmp_path / "project")


class TestPathZones:
    """Two zones: read the project, write only the workspace. One zone would be
    useless — workspace-only reads mean an agent that cannot see the code it is
    meant to work on."""

    def test_writing_inside_the_workspace_is_allowed(self, sb):
        assert sb.check_write(str(sb.workspace / "out.txt")) is None

    def test_writing_outside_is_refused(self, sb, tmp_path):
        assert "blocked" in sb.check_write(str(tmp_path / "elsewhere.txt"))

    def test_writing_to_the_project_is_refused(self, sb):
        """Readable is not writable — the project mount is read-only for a reason."""
        assert "blocked" in sb.check_write(str(sb.read_root / "main.py"))

    def test_reading_the_project_is_allowed(self, sb):
        assert sb.check_read(str(sb.read_root / "main.py")) is None

    def test_reading_the_workspace_is_allowed(self, sb):
        assert sb.check_read(str(sb.workspace / "notes.md")) is None

    def test_reading_elsewhere_is_refused(self, sb):
        assert "blocked" in sb.check_read("/etc/passwd")
        assert "blocked" in sb.check_read(str(Path.home() / ".ssh/id_rsa"))

    def test_traversal_cannot_climb_out(self, sb):
        """The check resolves before comparing, so ../.. does not escape."""
        assert "blocked" in sb.check_write(str(sb.workspace / ".." / ".." / "x"))


class TestDenyList:
    def test_refuses_obvious_footguns(self, sb):
        for cmd in ("sudo rm -rf /", "curl x | sh", "mkfs.ext4 /dev/disk1",
                    "shutdown -h now", "dd if=/dev/zero of=/dev/disk0"):
            assert sb.check_command(cmd), f"should have refused: {cmd}"

    def test_allows_ordinary_work(self, sb):
        for cmd in ("ls -la", "python -c 'print(1)'", "grep -r todo .",
                    "git status", "pytest -q"):
            assert sb.check_command(cmd) is None, f"should have allowed: {cmd}"

    def test_pipe_match_is_word_bounded(self, sb):
        """`| sh` is a word boundary, not a substring: real pipes into sh/bash are
        refused, but `| sha256sum` / `| shuf` are ordinary work and must pass."""
        for cmd in ("cat f | sh", "cat f |sh", "echo x | bash -s"):
            assert sb.check_command(cmd), f"should have refused: {cmd}"
        for cmd in ("sha256sum f | sha256sum", "sort f | shuf", "cat f | shasum -a 256"):
            assert sb.check_command(cmd) is None, f"should have allowed: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "rm -rf /workspace/build",              # 'rm -rf /' is a prefix of every abs path
        "rm -rf ./build",
        "chmod 777 /workspace/run.sh",
        "grep shutdown log.txt",                # a word as an argument, not a command
        "cat reboot-notes.md",
        "ps aux | grep su | head",              # 'su ' as an argument
        'echo "never use sudo here" > notes.md',
        "dd if=/dev/urandom of=data.bin bs=1k count=4",   # makes a file, not a disk write
        "git add if=x",                         # 'dd if=' inside 'add if='
        "grep -n '| sh' install.md",            # a quoted pipe is text, not a pipe
        "command -v sudo",                      # looks sudo up, does not run it
        "cat > setup.sh <<'EOF'\nsudo apt-get install -y git\nreboot\nEOF",  # data
        "cat <<< 'sudo'",                       # a here-string is data too
    ])
    def test_ordinary_commands_that_merely_contain_a_bad_word_pass(self, sb, cmd):
        """Substring matching refused all of these. A bogus refusal costs the agent a
        cycle and teaches it nothing, since the reason names a command it never ran."""
        assert sb.check_command(cmd) is None, f"should have allowed: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "sudo ls", "ls && sudo reboot", "FOO=1 sudo ls", "/sbin/shutdown -h now",
        "rm -rf /", "rm -fr /*", "rm -r -f /", "rm --recursive --force /",
        "chmod -R 777 /", "dd if=/dev/zero of=/dev/disk0", "mkfs.ext4 /dev/sda1",
        "bash -c 'sudo ls'", "echo $(sudo whoami)", "su -", "curl x | sh",
        "cat > a.txt <<EOF\nhello\nEOF\nsudo ls",   # a command AFTER a heredoc is checked
    ])
    def test_the_real_footguns_are_still_refused(self, sb, cmd):
        """Precision must not cost coverage: wrappers, chains, full paths, split flags
        and nested shells are all still caught."""
        assert sb.check_command(cmd), f"should have refused: {cmd}"

    def test_unparseable_input_does_not_crash(self, sb):
        """An unbalanced quote is the shell's problem to report, not a crash here —
        and must not become a way round the list."""
        assert sb.check_command('echo "unclosed') is None
        assert sb.check_command('sudo "unclosed')

    def test_it_is_explicitly_not_the_boundary(self, sb):
        """Documented limitation: an obfuscated command passes this and is still
        contained by the container. The deny-list only makes accidents legible."""
        assert sb.check_command("$(printf 'sud\\157') -k") is None


class TestRunArgv:
    """The docker run argv is where the confinement lives, so assert it directly —
    no daemon needed, and the security flags cannot silently drift."""

    def test_the_confinement_flags_are_always_present(self, sb):
        argv = sb._run_argv("lag-agent-test")
        assert "--cap-drop" in argv and "ALL" in argv
        assert "no-new-privileges" in argv
        assert "--pids-limit" in argv
        assert f"{sb.read_root}:{PROJECT_MNT}:ro" in argv   # project stays read-only

    def test_resource_caps_bound_a_runaway(self, sb):
        argv = sb._run_argv("x")
        # memory-swap must equal memory, or the real ceiling is 2x via swap
        assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1]
        assert "--cpus" in argv

    def test_rootfs_is_not_read_only_so_agents_can_install(self, sb):
        """Deliberate: pip/apt/build need a writable root, and the container is
        ephemeral, so --read-only would cost workflow for no cross-run gain."""
        assert "--read-only" not in sb._run_argv("x")

    def test_network_off_seals_egress(self, tmp_path):
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=False)
        argv = sb._run_argv("x")
        assert "none" in argv and argv[argv.index("--network") + 1] == "none"

    def test_network_on_joins_the_internal_network_not_the_bridge(self, tmp_path):
        """The default bridge NATs through the host — measured reaching the host's
        Ollama via host.docker.internal. Network on means the agent's own --internal
        network, whose only way out is the egress proxy."""
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=True)
        argv = sb._run_argv("lag-agent-abc123")
        assert argv[argv.index("--network") + 1] == "lag-net-abc123"
        assert "bridge" not in argv and "host" not in argv
        # and never a published port: egress is outbound only
        assert "-p" not in argv and "--publish" not in argv

    def test_network_on_points_every_proxy_variable_at_the_proxy(self, tmp_path):
        """curl reads only lowercase http_proxy; pip reads either. Miss one spelling and
        that tool silently has no network rather than a filtered one."""
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=True)
        argv = sb._run_argv("lag-agent-abc123")
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            assert f"{var}=http://lag-egress-abc123:3128" in argv, var
        assert "NO_PROXY=" in argv and "no_proxy=" in argv

    def test_the_proxy_is_hardened_and_sees_nothing_of_the_host(self, tmp_path):
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=True)
        argv = sb._proxy_argv("lag-egress-x")
        assert "--cap-drop" in argv and "ALL" in argv
        assert "no-new-privileges" in argv and "--read-only" in argv
        assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1]
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        assert len(mounts) == 2 and all(m.endswith(":ro") for m in mounts)
        assert not any(str(sb.workspace) in m or str(sb.read_root) + ":" in m
                       for m in mounts), "the proxy must not see the project or workspace"


@pytest.mark.skipif(not available()[0], reason="needs a running container runtime")
class TestRealContainment:
    """The only test that proves anything about the boundary itself."""

    def test_it_actually_contains(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "readme.txt").write_text("hello from the project")
        sb = Sandbox(tmp_path / "ws", read_root=project, network=False)  # sealed on purpose
        assert sb.start() is None, "container should start"
        try:
            p = SandboxedProvider(FakeInner(), sb)

            # works inside the workspace
            assert "ok" in p.call("run_shell", {"command": "echo ok > f && cat f"})
            # can read the project
            assert "hello from the project" in p.call(
                "run_shell", {"command": "cat /project/readme.txt"})
            # cannot write to the project
            assert "Read-only" in p.call(
                "run_shell", {"command": "touch /project/escaped.txt"})
            assert not (project / "escaped.txt").exists()
            # cannot see the host filesystem
            assert "No such file" in p.call(
                "run_shell", {"command": f"ls {tmp_path}"})
            # no network
            assert "no network" in p.call(
                "run_shell", {"command": "getent hosts example.com || echo no network"})
        finally:
            sb.stop()

    def test_stop_removes_the_container(self, tmp_path):
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path)
        assert sb.start() is None
        name = sb.container
        sb.stop()
        assert sb.container is None
        import subprocess
        from lag.sandbox import _env_for
        r = subprocess.run([sb.cli, "ps", "-a", "--filter", f"name={name}", "-q"],
                           capture_output=True, text=True, env=_env_for(sb.cli))
        assert r.stdout.strip() == "", "container should be gone"


@pytest.mark.skipif(not available()[0], reason="needs a running container runtime")
class TestRealEgress:
    """The attack from the review, against real containers: with network on, the
    agent's shell reached the host's Ollama through host.docker.internal."""

    PY = ("import sys, urllib.request as u\n"
          "try:\n"
          "    r = u.urlopen(sys.argv[1], timeout=8); print('REACHED', r.status)\n"
          "except Exception as e:\n"
          "    print('REFUSED', str(e)[:120])\n")

    def _fetch(self, p, url):
        import shlex
        return p.call("run_shell", {"command": f"python -c {shlex.quote(self.PY)} {url}"})

    def test_the_host_lan_and_tailnet_are_refused_but_the_internet_is_not(self, tmp_path):
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=True)
        assert sb.start() is None, "sandbox with egress should start"
        try:
            p = SandboxedProvider(FakeInner(), sb)
            for url in ("http://host.docker.internal:11434/api/version",
                        "http://192.168.65.254:11434/api/version",
                        "http://127.0.0.1:11434/",
                        "http://100.100.100.100/"):
                out = self._fetch(p, url)
                assert "REACHED" not in out, (url, out)
            # past the proxy there is no route at all: raw sockets ignore HTTP_PROXY
            raw = p.call("run_shell", {"command":
                "python -c \"import socket; socket.create_connection(('1.1.1.1', 80), 5)\" "
                "2>&1 | tail -1"})
            assert "unreachable" in raw.lower() or "timed out" in raw.lower(), raw
            # and the public internet still works, which is the point of network on
            assert "REACHED 200" in self._fetch(p, "https://pypi.org/simple/pip/")
        finally:
            sb.stop()

    def test_stop_removes_the_proxy_and_the_network(self, tmp_path):
        import subprocess
        from lag.sandbox import _env_for
        sb = Sandbox(tmp_path / "ws", read_root=tmp_path, network=True)
        assert sb.start() is None
        proxy, net = sb.proxy, sb.net
        sb.stop()
        env = _env_for(sb.cli)
        ps = subprocess.run([sb.cli, "ps", "-a", "-q", "--filter", f"name={proxy}"],
                            capture_output=True, text=True, env=env)
        nets = subprocess.run([sb.cli, "network", "ls", "-q", "--filter", f"name={net}"],
                              capture_output=True, text=True, env=env)
        assert ps.stdout.strip() == "" and nets.stdout.strip() == ""


class TestPathTranslation:
    """The agent lives in the container's namespace; the file tools run on the host.

    Without translation an agent is stuck: `pwd` says /workspace, which does not exist
    on the host, and a bare relative path resolves against the repo rather than the
    workspace — so every path it knows is rejected. Observed exactly that, looping to
    its cycle cap writing files that went nowhere.
    """

    def _p(self, tmp_path):
        project = tmp_path / "project"; project.mkdir()
        sb = Sandbox(tmp_path / "ws", read_root=project)
        return SandboxedProvider(FakeInner(), sb), sb

    def test_container_workspace_path_maps_to_the_host_workspace(self, tmp_path):
        p, sb = self._p(tmp_path)
        assert p._translate("/workspace/out.txt") == str(sb.workspace / "out.txt")

    def test_container_project_path_maps_to_the_host_project(self, tmp_path):
        p, sb = self._p(tmp_path)
        assert p._translate("/project/main.py") == str(sb.read_root / "main.py")

    def test_a_relative_path_means_the_workspace(self, tmp_path):
        """The shell starts in /workspace, so that is what relative means to the agent."""
        p, sb = self._p(tmp_path)
        assert p._translate("notes.md") == str(sb.workspace / "notes.md")

    def test_a_relative_path_that_only_exists_in_the_project_reads_from_there(self, tmp_path):
        p, sb = self._p(tmp_path)
        (sb.read_root / "main.py").write_text("x")
        assert p._translate("main.py") == str(sb.read_root / "main.py")

    def test_the_workspace_wins_when_both_exist(self, tmp_path):
        p, sb = self._p(tmp_path)
        (sb.read_root / "notes.md").write_text("project copy")
        (sb.workspace / "notes.md").write_text("agent copy")
        assert p._translate("notes.md") == str(sb.workspace / "notes.md")

    @pytest.mark.parametrize("name", ["README.md", "src/main.py"])
    def test_a_relative_write_lands_in_the_workspace_even_if_the_project_has_it(
            self, tmp_path, name):
        """The project-fallback is a READ rule. Applied to writes it sent
        write_file('README.md') to the read-only project, the zone check refused it,
        and the agent retried the same call until its cycle cap."""
        p, sb = self._p(tmp_path)
        (sb.read_root / name).parent.mkdir(parents=True, exist_ok=True)
        (sb.read_root / name).write_text("project copy")
        assert p.call("write_file", {"path": name, "content": "x"}) == "inner ran write_file"
        assert p.inner.called[-1][1]["path"] == str(sb.workspace / name)

    def test_read_edit_reread_behaves_like_copy_on_write(self, tmp_path):
        """End to end through the real file tools: the agent reads the project's file,
        writes its edit, then reads back ITS copy. The project is never touched."""
        from lag.tools import LocalToolProvider
        project = tmp_path / "project"; project.mkdir()
        (project / "README.md").write_text("original")
        sb = Sandbox(tmp_path / "ws", read_root=project)
        p = SandboxedProvider(LocalToolProvider(), sb)

        assert p.call("read_file", {"path": "README.md"}) == "original"
        assert "blocked" not in p.call("write_file", {"path": "README.md", "content": "edited"})
        assert p.call("read_file", {"path": "README.md"}) == "edited"
        assert (project / "README.md").read_text() == "original"
        assert (sb.workspace / "README.md").read_text() == "edited"

    def test_a_translated_write_is_then_allowed(self, tmp_path):
        """Translation and the zone check have to agree, or the agent is still stuck."""
        p, sb = self._p(tmp_path)
        assert p.call("write_file", {"path": "/workspace/a.txt", "content": "x"}) \
            == "inner ran write_file"

    def test_translation_does_not_open_an_escape(self, tmp_path):
        p, sb = self._p(tmp_path)
        assert "blocked" in p.call(
            "write_file", {"path": "/workspace/../../../etc/hosts", "content": "x"})
        assert "blocked" in p.call("write_file", {"path": "/etc/hosts", "content": "x"})


class TestRuntimeDiscovery:
    """Docker Desktop's symlinks break in a specific, common way.

    Running it from the mounted disk image leaves /usr/local/bin/docker pointing at
    /Volumes/Docker/..., which dangles the moment the image is ejected. `which` finds
    the name, the path exists as a link, and every call fails. Discovery has to look
    past it rather than stopping at the first thing named "docker".
    """

    def test_a_dangling_symlink_is_not_treated_as_a_runtime(self, tmp_path, monkeypatch):
        import lag.sandbox as sbmod

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        dangling = fake_bin / "docker"
        dangling.symlink_to("/Volumes/Docker/Docker.app/Contents/Resources/bin/docker")
        assert dangling.is_symlink() and not dangling.exists()

        real = tmp_path / "app" / "docker"
        real.parent.mkdir()
        real.write_text("#!/bin/sh\necho hi\n")
        real.chmod(0o755)

        monkeypatch.setattr(sbmod, "_CANDIDATES", [str(dangling), str(real)])
        assert sbmod.find_runtime() == str(real)

    def test_no_runtime_at_all_is_reported_clearly(self, monkeypatch):
        import lag.sandbox as sbmod

        monkeypatch.setattr(sbmod, "_CANDIDATES", ["/nope/docker"])
        ok, why = sbmod.available()
        assert not ok and "no container runtime" in why

    def test_a_present_cli_with_a_dead_daemon_says_so(self, tmp_path, monkeypatch):
        """The message has to distinguish 'not installed' from 'not started' — they
        have completely different fixes."""
        import lag.sandbox as sbmod

        cli = tmp_path / "docker"
        cli.write_text("#!/bin/sh\nexit 1\n")
        cli.chmod(0o755)
        monkeypatch.setattr(sbmod, "_CANDIDATES", [str(cli)])
        ok, why = sbmod.available()
        assert not ok and "daemon is not running" in why


class TestSearchConfinement:
    """find_files/grep_files take a SECOND argument that picks files: the glob.

    Checking only `path` approved the starting directory while the glob walked out of
    it — grep_files(path="/project", glob="../../../etc/passwd") returned /etc/passwd.
    A symlink the agent plants in its workspace is the same escape by another route.
    """

    def _p(self, tmp_path):
        from lag.tools import LocalToolProvider
        project = tmp_path / "project"; project.mkdir()
        (project / "app.py").write_text("SECRET_IN_PROJECT = 1\n")
        outside = tmp_path / "outside"; outside.mkdir()
        (outside / "id_rsa").write_text("PRIVATE KEY\n")
        sb = Sandbox(tmp_path / "ws", read_root=project)
        sb.workspace.mkdir(parents=True, exist_ok=True)
        return SandboxedProvider(LocalToolProvider(), sb), sb, outside

    def test_grep_glob_cannot_climb_out(self, tmp_path):
        p, sb, outside = self._p(tmp_path)
        out = p.call("grep_files", {"pattern": "PRIVATE", "path": str(sb.read_root),
                                    "glob": "../outside/*"})
        assert "PRIVATE KEY" not in out and "blocked" in out

    def test_find_pattern_cannot_climb_out(self, tmp_path):
        p, sb, outside = self._p(tmp_path)
        out = p.call("find_files", {"pattern": "../outside/*", "path": str(sb.read_root)})
        assert "id_rsa" not in out and "blocked" in out

    def test_absolute_glob_is_refused(self, tmp_path):
        p, sb, outside = self._p(tmp_path)
        out = p.call("grep_files", {"pattern": "x", "path": str(sb.read_root),
                                    "glob": str(outside / "*")})
        assert "blocked" in out

    def test_a_planted_symlink_is_not_followed_out(self, tmp_path):
        """The container can `ln -s` anything into the workspace; the host-side search
        must not follow it to the real target."""
        p, sb, outside = self._p(tmp_path)
        (sb.workspace / "keys").symlink_to(outside)
        # `**` happens not to descend into symlinked dirs, but naming the link does.
        for glob in ("**/*", "keys/*"):
            out = p.call("grep_files", {"pattern": "PRIVATE", "path": str(sb.workspace),
                                        "glob": glob})
            assert "PRIVATE KEY" not in out, glob
            out = p.call("find_files", {"pattern": glob, "path": str(sb.workspace)})
            assert "id_rsa" not in out, glob

    def test_ordinary_searches_still_work(self, tmp_path):
        p, sb, outside = self._p(tmp_path)
        assert "app.py" in p.call("find_files", {"pattern": "*.py", "path": "/project"})
        assert "SECRET_IN_PROJECT" in p.call(
            "grep_files", {"pattern": "SECRET", "path": "/project", "glob": "**/*.py"})
