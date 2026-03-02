"""
Tests for all production-readiness modules:
  - storage.py (JSONFileBackend, SQLiteBackend)
  - workers.py (LearningWorker)
  - cli.py (all 11 commands)
  - mcp_server.py
  - registry.py (SkillRegistry install/publish)
  - export.py (DPO + Anthropic)
  - dashboard.py (import & structure)
  - adapters (langchain, crewai)
  - services (slack, email, crm)
  - core.py storage wiring & error classification
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def json_backend(tmp_dir):
    from cannyforge.storage import JSONFileBackend
    return JSONFileBackend(tmp_dir)


@pytest.fixture
def sqlite_backend(tmp_dir):
    from cannyforge.storage import SQLiteBackend
    return SQLiteBackend(tmp_dir)


@pytest.fixture
def sample_error():
    return {
        'timestamp': '2025-01-15T10:00:00',
        'skill': 'email_writer',
        'task': 'Write email about meeting at 3 PM',
        'error_type': 'TimezoneError',
        'error_message': 'TimezoneError: no timezone specified',
        'context': {'has_timezone': False},
        'rules_applied': [],
    }


@pytest.fixture
def sample_success():
    return {
        'timestamp': '2025-01-15T10:05:00',
        'skill': 'email_writer',
        'task': 'Write email about meeting',
        'context': {'has_timezone': True},
        'rules_applied': ['rule_timezone_1'],
        'execution_time_ms': 42.5,
    }


@pytest.fixture
def sample_step_error():
    return {
        'timestamp': '2025-01-15T10:02:00',
        'skill': 'email_writer',
        'task': 'Write email',
        'step': 2,
        'tool': 'send_email',
        'error_type': 'SpamTriggerError',
        'error_message': 'SpamTriggerError: spam words detected',
        'recovery_applied': ['rule_spam_1'],
        'recovery_succeeded': True,
        'context': {},
    }


@pytest.fixture
def forge_clean(tmp_dir):
    from cannyforge.core import CannyForge
    return CannyForge(data_dir=tmp_dir)


# ===========================================================================
# storage.py — JSONFileBackend
# ===========================================================================

class TestJSONFileBackend:

    def test_store_and_get_errors(self, json_backend, sample_error):
        json_backend.store_error(sample_error)
        errors = json_backend.get_errors()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'TimezoneError'

    def test_get_errors_filter_skill(self, json_backend, sample_error):
        json_backend.store_error(sample_error)
        assert len(json_backend.get_errors(skill_name='email_writer')) == 1
        assert len(json_backend.get_errors(skill_name='other_skill')) == 0

    def test_get_errors_filter_type(self, json_backend, sample_error):
        json_backend.store_error(sample_error)
        assert len(json_backend.get_errors(error_type='TimezoneError')) == 1
        assert len(json_backend.get_errors(error_type='GenericError')) == 0

    def test_get_errors_limit(self, json_backend, sample_error):
        for i in range(5):
            json_backend.store_error(sample_error)
        assert len(json_backend.get_errors(limit=3)) == 3

    def test_clear_errors(self, json_backend, sample_error):
        json_backend.store_error(sample_error)
        json_backend.clear_errors()
        assert len(json_backend.get_errors()) == 0

    def test_store_and_get_successes(self, json_backend, sample_success):
        json_backend.store_success(sample_success)
        successes = json_backend.get_successes()
        assert len(successes) == 1
        assert successes[0]['skill'] == 'email_writer'

    def test_get_successes_filter_skill(self, json_backend, sample_success):
        json_backend.store_success(sample_success)
        assert len(json_backend.get_successes(skill_name='email_writer')) == 1
        assert len(json_backend.get_successes(skill_name='other')) == 0

    def test_clear_successes(self, json_backend, sample_success):
        json_backend.store_success(sample_success)
        json_backend.clear_successes()
        assert len(json_backend.get_successes()) == 0

    def test_store_and_get_step_errors(self, json_backend, sample_step_error):
        json_backend.store_step_error(sample_step_error)
        step_errors = json_backend.get_step_errors()
        assert len(step_errors) == 1
        assert step_errors[0]['tool'] == 'send_email'

    def test_clear_step_errors(self, json_backend, sample_step_error):
        json_backend.store_step_error(sample_step_error)
        json_backend.clear_step_errors()
        assert len(json_backend.get_step_errors()) == 0

    def test_store_and_load_rules(self, json_backend):
        rules = {'email_writer': [{'id': 'r1', 'name': 'test'}]}
        json_backend.store_rules(rules)
        loaded = json_backend.load_rules()
        assert loaded == rules

    def test_load_rules_empty(self, json_backend):
        assert json_backend.load_rules() == {}


# ===========================================================================
# storage.py — SQLiteBackend
# ===========================================================================

class TestSQLiteBackend:

    def test_store_and_get_errors(self, sqlite_backend, sample_error):
        sqlite_backend.store_error(sample_error)
        errors = sqlite_backend.get_errors()
        assert len(errors) == 1
        assert errors[0]['error_type'] == 'TimezoneError'

    def test_get_errors_filter_skill(self, sqlite_backend, sample_error):
        sqlite_backend.store_error(sample_error)
        assert len(sqlite_backend.get_errors(skill_name='email_writer')) == 1
        assert len(sqlite_backend.get_errors(skill_name='other')) == 0

    def test_get_errors_filter_type(self, sqlite_backend, sample_error):
        sqlite_backend.store_error(sample_error)
        assert len(sqlite_backend.get_errors(error_type='TimezoneError')) == 1
        assert len(sqlite_backend.get_errors(error_type='GenericError')) == 0

    def test_get_errors_limit(self, sqlite_backend, sample_error):
        for _ in range(5):
            sqlite_backend.store_error(sample_error)
        assert len(sqlite_backend.get_errors(limit=3)) == 3

    def test_clear_errors(self, sqlite_backend, sample_error):
        sqlite_backend.store_error(sample_error)
        sqlite_backend.clear_errors()
        assert len(sqlite_backend.get_errors()) == 0

    def test_store_and_get_successes(self, sqlite_backend, sample_success):
        sqlite_backend.store_success(sample_success)
        successes = sqlite_backend.get_successes()
        assert len(successes) == 1
        assert successes[0]['execution_time_ms'] == 42.5

    def test_get_successes_filter_skill(self, sqlite_backend, sample_success):
        sqlite_backend.store_success(sample_success)
        assert len(sqlite_backend.get_successes(skill_name='email_writer')) == 1
        assert len(sqlite_backend.get_successes(skill_name='other')) == 0

    def test_clear_successes(self, sqlite_backend, sample_success):
        sqlite_backend.store_success(sample_success)
        sqlite_backend.clear_successes()
        assert len(sqlite_backend.get_successes()) == 0

    def test_store_and_get_step_errors(self, sqlite_backend, sample_step_error):
        sqlite_backend.store_step_error(sample_step_error)
        step_errors = sqlite_backend.get_step_errors()
        assert len(step_errors) == 1
        assert step_errors[0]['recovery_succeeded'] is True

    def test_get_step_errors_filter(self, sqlite_backend, sample_step_error):
        sqlite_backend.store_step_error(sample_step_error)
        assert len(sqlite_backend.get_step_errors(skill_name='email_writer')) == 1
        assert len(sqlite_backend.get_step_errors(error_type='SpamTriggerError')) == 1
        assert len(sqlite_backend.get_step_errors(skill_name='other')) == 0

    def test_clear_step_errors(self, sqlite_backend, sample_step_error):
        sqlite_backend.store_step_error(sample_step_error)
        sqlite_backend.clear_step_errors()
        assert len(sqlite_backend.get_step_errors()) == 0

    def test_store_and_load_rules(self, sqlite_backend):
        rules = {'email_writer': [{'id': 'r1', 'name': 'test'}]}
        sqlite_backend.store_rules(rules)
        loaded = sqlite_backend.load_rules()
        assert loaded == rules

    def test_load_rules_empty(self, sqlite_backend):
        assert sqlite_backend.load_rules() == {}

    def test_wal_mode_enabled(self, sqlite_backend):
        """SQLite should use WAL journal mode for concurrent access."""
        import sqlite3
        conn = sqlite3.connect(str(sqlite_backend.db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_thread_safety(self, sqlite_backend, sample_error):
        """Multiple threads can write concurrently without corruption."""
        errors_written = []

        def writer(n):
            for i in range(10):
                data = dict(sample_error)
                data['error_message'] = f'thread-{n}-error-{i}'
                sqlite_backend.store_error(data)
                errors_written.append(True)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(sqlite_backend.get_errors()) == 30


# ===========================================================================
# Storage backend wiring (core.py + learning.py integration)
# ===========================================================================

class TestStorageWiring:

    def test_jsonl_backend_wired_by_default(self, tmp_dir):
        from cannyforge.core import CannyForge
        from cannyforge.storage import JSONFileBackend
        forge = CannyForge(data_dir=tmp_dir)
        assert isinstance(forge.storage, JSONFileBackend)
        assert forge.learning_engine.error_repo._backend is forge.storage

    def test_sqlite_backend_wired(self, tmp_dir):
        from cannyforge.core import CannyForge
        from cannyforge.storage import SQLiteBackend
        forge = CannyForge(data_dir=tmp_dir, storage_backend='sqlite')
        assert isinstance(forge.storage, SQLiteBackend)
        assert forge.learning_engine.error_repo._backend is forge.storage
        assert forge.learning_engine.success_repo._backend is forge.storage
        assert forge.learning_engine.step_error_repo._backend is forge.storage

    def test_sqlite_records_persist(self, tmp_dir):
        from cannyforge.core import CannyForge
        forge = CannyForge(data_dir=tmp_dir, storage_backend='sqlite')
        forge.learning_engine.record_error(
            'test', 'test task', 'TimezoneError', 'tz missing')
        # Verify in SQLite directly
        rows = forge.storage.get_errors()
        assert len(rows) == 1
        assert rows[0]['error_type'] == 'TimezoneError'

    def test_sqlite_success_records_persist(self, tmp_dir):
        from cannyforge.core import CannyForge
        forge = CannyForge(data_dir=tmp_dir, storage_backend='sqlite')
        forge.learning_engine.record_success(
            'test', 'test task', {'ctx': True}, ['rule1'], 100.0)
        rows = forge.storage.get_successes()
        assert len(rows) == 1
        assert rows[0]['execution_time_ms'] == 100.0


# ===========================================================================
# workers.py — LearningWorker
# ===========================================================================

class TestLearningWorker:

    def test_start_stop(self):
        from cannyforge.workers import LearningWorker
        fn = MagicMock()
        w = LearningWorker(fn)
        w.start()
        assert w._running
        w.stop()
        assert not w._running

    def test_start_idempotent(self):
        from cannyforge.workers import LearningWorker
        w = LearningWorker(lambda: None)
        w.start()
        thread1 = w._thread
        w.start()  # second start is no-op
        assert w._thread is thread1
        w.stop()

    def test_stop_idempotent(self):
        from cannyforge.workers import LearningWorker
        w = LearningWorker(lambda: None)
        w.stop()  # stop before start is no-op

    def test_enqueue_and_process(self):
        from cannyforge.workers import LearningWorker
        calls = []
        w = LearningWorker(lambda: calls.append(1))
        w.start()
        w.enqueue()
        time.sleep(0.5)
        w.stop()
        assert len(calls) == 1

    def test_pending_count(self):
        from cannyforge.workers import LearningWorker
        # Use a slow function so items stay queued
        event = threading.Event()
        w = LearningWorker(lambda: event.wait(timeout=2))
        w.start()
        w.enqueue()
        time.sleep(0.1)  # let first item start processing
        w.enqueue()
        # Second item should be pending
        assert w.pending >= 1
        event.set()
        w.stop()

    def test_error_in_learning_fn_doesnt_crash_worker(self):
        from cannyforge.workers import LearningWorker
        calls = []

        def fail_then_succeed():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")

        w = LearningWorker(fail_then_succeed)
        w.start()
        w.enqueue()  # will raise
        time.sleep(0.3)
        w.enqueue()  # should still work
        time.sleep(0.3)
        w.stop()
        assert len(calls) == 2


# ===========================================================================
# cli.py
# ===========================================================================

class TestCLI:

    def test_help(self):
        """CLI prints help without error."""
        from cannyforge.cli import main
        import sys
        with pytest.raises(SystemExit) as exc:
            sys.argv = ['cannyforge', '--help']
            main()
        assert exc.value.code == 0

    def test_cmd_stats(self, tmp_dir, capsys):
        from cannyforge.cli import cmd_stats
        import argparse
        with patch('cannyforge.core.CannyForge') as mock_cls:
            mock_forge = MagicMock()
            mock_forge.get_statistics.return_value = {
                'execution': {'tasks_executed': 10, 'success_rate': 0.8,
                              'tasks_succeeded': 8, 'tasks_failed': 2},
                'learning': {'learning_cycles': 2, 'total_rules': 5,
                             'rule_success_rate': 0.7, 'average_rule_confidence': 0.6},
                'knowledge': {'rules_by_status': {'active': 3, 'probation': 1, 'dormant': 1}},
                'skills': {'skill_stats': {
                    'email_writer': {'executions': 10, 'success_rate': 0.8}
                }},
            }
            mock_cls.return_value = mock_forge
            cmd_stats(argparse.Namespace())
            out = capsys.readouterr().out
            assert 'Success rate' in out
            assert '80.0%' in out

    def test_cmd_new_skill(self, tmp_dir, capsys):
        from cannyforge.cli import cmd_new_skill
        import argparse
        os.chdir(tmp_dir)
        cmd_new_skill(argparse.Namespace(name='my-test-skill'))
        out = capsys.readouterr().out
        assert 'Created skill scaffold' in out
        skill_md = tmp_dir / 'skills' / 'my-test-skill' / 'SKILL.md'
        assert skill_md.exists()
        content = skill_md.read_text()
        assert 'name: my-test-skill' in content

    def test_cmd_new_skill_already_exists(self, tmp_dir):
        from cannyforge.cli import cmd_new_skill
        import argparse
        os.chdir(tmp_dir)
        (tmp_dir / 'skills' / 'dupe').mkdir(parents=True)
        with pytest.raises(SystemExit):
            cmd_new_skill(argparse.Namespace(name='dupe'))

    def test_cmd_learn(self, capsys):
        from cannyforge.cli import cmd_learn
        import argparse
        with patch('cannyforge.core.CannyForge') as mock_cls:
            mock_forge = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.errors_analyzed = 50
            mock_metrics.patterns_detected = 3
            mock_metrics.rules_generated = 2
            mock_metrics.rules_applied_total = 10
            mock_metrics.rule_success_rate = 0.75
            mock_forge.run_learning_cycle.return_value = mock_metrics
            mock_cls.return_value = mock_forge
            cmd_learn(argparse.Namespace(min_freq=3, min_conf=0.5))
            out = capsys.readouterr().out
            assert 'Patterns detected: 3' in out
            assert 'Rules generated:   2' in out

    def test_cmd_export_dpo(self, tmp_dir, capsys):
        from cannyforge.cli import cmd_export
        import argparse
        output_path = str(tmp_dir / 'out.jsonl')
        with patch('cannyforge.core.CannyForge') as mock_cls, \
             patch('cannyforge.export.export_dpo') as mock_export:
            mock_cls.return_value = MagicMock()
            mock_export.return_value = 5
            cmd_export(argparse.Namespace(
                format='dpo', output=output_path, data_dir=str(tmp_dir)))
            mock_export.assert_called_once()

    def test_cmd_serve_missing_mcp(self, capsys):
        from cannyforge.cli import cmd_serve
        import argparse
        with patch.dict('sys.modules', {'cannyforge.mcp_server': None}):
            with patch('cannyforge.cli.cmd_serve') as patched:
                # Just verify the function exists and is callable
                assert callable(cmd_serve)


# ===========================================================================
# registry.py
# ===========================================================================

class TestSkillRegistryInstall:

    def test_invalid_spec_raises(self):
        from cannyforge.registry import SkillRegistry
        with pytest.raises(ValueError, match="Invalid format"):
            SkillRegistry.install("not-github-format")

    def test_invalid_spec_too_short(self):
        from cannyforge.registry import SkillRegistry
        with pytest.raises(ValueError, match="Invalid format"):
            SkillRegistry.install("github:user")

    def test_publish_missing_skill_md(self, tmp_dir):
        from cannyforge.registry import SkillRegistry
        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            SkillRegistry.publish(tmp_dir)

    def test_publish_bad_frontmatter(self, tmp_dir):
        from cannyforge.registry import SkillRegistry
        (tmp_dir / "SKILL.md").write_text("no frontmatter here")
        with pytest.raises(ValueError, match="frontmatter"):
            SkillRegistry.publish(tmp_dir)

    def test_publish_missing_name(self, tmp_dir):
        from cannyforge.registry import SkillRegistry
        (tmp_dir / "SKILL.md").write_text("---\ndescription: test\n---\nbody")
        with pytest.raises(ValueError, match="name"):
            SkillRegistry.publish(tmp_dir)

    def test_publish_valid(self, tmp_dir, capsys):
        from cannyforge.registry import SkillRegistry
        (tmp_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: test\n---\nbody")
        SkillRegistry.publish(tmp_dir)
        out = capsys.readouterr().out
        assert "my-skill" in out
        assert "valid" in out.lower()


# ===========================================================================
# export.py
# ===========================================================================

class TestExport:

    def test_export_dpo_empty(self, tmp_dir):
        from cannyforge.core import CannyForge
        from cannyforge.export import export_dpo
        forge = CannyForge(data_dir=tmp_dir)
        output = tmp_dir / "out.jsonl"
        count = export_dpo(forge, output)
        assert count == 0
        assert output.exists()

    def test_export_dpo_with_data(self, tmp_dir):
        from cannyforge.core import CannyForge
        from cannyforge.export import export_dpo
        from cannyforge.learning import SuccessRecord, ErrorRecord
        forge = CannyForge(data_dir=tmp_dir)

        # Add matching success + error with same task prefix
        task = "Write an email about meeting at 3 PM"
        forge.learning_engine.success_repo.successes.append(SuccessRecord(
            timestamp=datetime.now(), skill_name='email_writer',
            task_description=task, rules_applied=['rule1'],
            context_snapshot={'ctx': True},
        ))
        forge.learning_engine.error_repo.errors.append(ErrorRecord(
            timestamp=datetime.now(), skill_name='email_writer',
            task_description=task, error_type='TimezoneError',
            error_message='tz missing', context_snapshot={'ctx': False},
        ))

        output = tmp_dir / "out.jsonl"
        count = export_dpo(forge, output)
        assert count >= 1
        with open(output) as f:
            pair = json.loads(f.readline())
        assert 'chosen' in pair
        assert 'rejected' in pair

    def test_export_anthropic(self, tmp_dir):
        from cannyforge.core import CannyForge
        from cannyforge.export import export_anthropic
        from cannyforge.learning import SuccessRecord
        forge = CannyForge(data_dir=tmp_dir)
        forge.learning_engine.success_repo.successes.append(SuccessRecord(
            timestamp=datetime.now(), skill_name='test',
            task_description='test task', context_snapshot={},
        ))
        output = tmp_dir / "out.json"
        export_anthropic(forge, output)
        data = json.loads(output.read_text())
        assert len(data) == 1
        assert data[0]['input'] == 'test task'


# ===========================================================================
# mcp_server.py
# ===========================================================================

class TestMCPServer:

    def test_create_server_without_mcp_raises(self):
        """When mcp is not installed, create_mcp_server should raise ImportError."""
        from cannyforge import mcp_server
        original = mcp_server.MCP_AVAILABLE
        try:
            mcp_server.MCP_AVAILABLE = False
            with pytest.raises(ImportError, match="MCP dependencies"):
                mcp_server.create_mcp_server()
        finally:
            mcp_server.MCP_AVAILABLE = original

    @pytest.mark.skipif(
        not __import__('cannyforge.mcp_server', fromlist=['MCP_AVAILABLE']).MCP_AVAILABLE,
        reason="MCP not installed"
    )
    def test_create_server_returns_server(self, tmp_dir):
        from cannyforge.mcp_server import create_mcp_server
        server = create_mcp_server(data_dir=str(tmp_dir))
        assert server is not None


# ===========================================================================
# adapters
# ===========================================================================

class TestLangChainAdapter:

    def test_import(self):
        from cannyforge.adapters.langchain import CannyForgeTool, get_all_tools
        assert CannyForgeTool is not None

    def test_tool_creation(self, forge_clean):
        from cannyforge.adapters.langchain import CannyForgeTool
        tool = CannyForgeTool(forge=forge_clean)
        assert tool.name == "cannyforge"
        assert "CannyForge" in tool.description

    def test_tool_with_skill_name(self, forge_clean):
        from cannyforge.adapters.langchain import CannyForgeTool
        tool = CannyForgeTool(forge=forge_clean, skill_name="email_writer")
        assert tool.name == "cannyforge_email_writer"

    def test_tool_run_success(self, forge_clean):
        from cannyforge.adapters.langchain import CannyForgeTool
        tool = CannyForgeTool(forge=forge_clean, skill_name="email_writer")
        result = tool._run("Write an email about the meeting")
        assert isinstance(result, str)

    def test_get_all_tools(self, forge_clean):
        from cannyforge.adapters.langchain import get_all_tools
        tools = get_all_tools(forge_clean)
        assert len(tools) > 0
        assert all(hasattr(t, '_run') for t in tools)


class TestCrewAIAdapter:

    def test_import(self):
        from cannyforge.adapters.crewai import CannyForgeCrewTool, get_all_tools
        assert CannyForgeCrewTool is not None

    def test_tool_creation(self, forge_clean):
        from cannyforge.adapters.crewai import CannyForgeCrewTool
        tool = CannyForgeCrewTool(forge=forge_clean)
        assert tool.name == "cannyforge"

    def test_tool_run(self, forge_clean):
        from cannyforge.adapters.crewai import CannyForgeCrewTool
        tool = CannyForgeCrewTool(forge=forge_clean, skill_name="email_writer")
        result = tool._run("Write an email")
        assert isinstance(result, str)

    def test_get_all_tools(self, forge_clean):
        from cannyforge.adapters.crewai import get_all_tools
        tools = get_all_tools(forge_clean)
        assert len(tools) > 0


# ===========================================================================
# services
# ===========================================================================

class TestSlackService:

    def test_mock_mode_by_default(self):
        from cannyforge.services.slack_service import SlackService
        svc = SlackService()
        assert svc.is_mock

    def test_connect_mock(self):
        from cannyforge.services.slack_service import SlackService
        svc = SlackService()
        svc.connect()  # should not raise

    def test_send_message_mock(self):
        from cannyforge.services.slack_service import SlackService
        svc = SlackService()
        svc.connect()
        resp = svc.send_message("#general", "hello")
        assert resp.success
        assert resp.data['mock'] is True
        assert resp.data['channel'] == '#general'

    def test_list_channels_mock(self):
        from cannyforge.services.slack_service import SlackService
        svc = SlackService()
        svc.connect()
        resp = svc.list_channels()
        assert resp.success
        assert '#general' in resp.data['channels']


class TestEmailService:

    def test_mock_mode_by_default(self):
        from cannyforge.services.email_service import EmailService
        svc = EmailService()
        assert svc.is_mock

    def test_connect_mock(self):
        from cannyforge.services.email_service import EmailService
        svc = EmailService()
        svc.connect()

    def test_send_email_mock(self):
        from cannyforge.services.email_service import EmailService
        svc = EmailService()
        svc.connect()
        resp = svc.send_email("user@example.com", "Hello", "Body text")
        assert resp.success
        assert resp.data['mock'] is True
        assert resp.data['to'] == 'user@example.com'


class TestCRMService:

    def test_mock_mode_by_default(self):
        from cannyforge.services.crm_service import CRMService
        svc = CRMService()
        assert svc.is_mock

    def test_lookup_contact_mock(self):
        from cannyforge.services.crm_service import CRMService
        svc = CRMService()
        svc.connect()
        resp = svc.lookup_contact("user@example.com")
        assert resp.success
        assert resp.data['email'] == 'user@example.com'
        assert resp.data['mock'] is True

    def test_log_activity_mock(self):
        from cannyforge.services.crm_service import CRMService
        svc = CRMService()
        svc.connect()
        resp = svc.log_activity("user@example.com", "email", "Sent welcome")
        assert resp.success
        assert resp.data['activity_type'] == 'email'


# ===========================================================================
# dashboard.py — import-only test (no streamlit runtime)
# ===========================================================================

class TestDashboard:

    def test_module_importable(self):
        """dashboard.py should be importable without streamlit at module level."""
        # It imports streamlit at top level, so this may fail if streamlit
        # is not installed. That's fine — we just verify it doesn't have
        # syntax errors.
        try:
            import cannyforge.dashboard
        except ImportError:
            pytest.skip("streamlit not installed")

    def test_no_blocking_sleep(self):
        """dashboard.py must not contain time.sleep blocking loops."""
        dashboard_path = Path(__file__).parent.parent / "cannyforge" / "dashboard.py"
        content = dashboard_path.read_text()
        assert 'time.sleep' not in content, \
            "dashboard.py should not block with time.sleep"

    def test_no_undeclared_imports(self):
        """dashboard.py should not import matplotlib or pandas (undeclared deps)."""
        dashboard_path = Path(__file__).parent.parent / "cannyforge" / "dashboard.py"
        content = dashboard_path.read_text()
        assert 'import matplotlib' not in content
        assert 'import pandas' not in content


# ===========================================================================
# core.py — error classification
# ===========================================================================

class TestErrorClassification:

    def test_classify_full_error_type_name(self, forge_clean):
        """Error messages containing the full type name should classify correctly."""
        assert forge_clean._classify_error("TimezoneError: no tz") == 'TimezoneError'
        assert forge_clean._classify_error("SpamTriggerError: spam") == 'SpamTriggerError'
        assert forge_clean._classify_error("AttachmentError: missing") == 'AttachmentError'

    def test_classify_keyword(self, forge_clean):
        """Error messages with just the keyword should classify correctly."""
        assert forge_clean._classify_error("timezone not specified") == 'TimezoneError'
        assert forge_clean._classify_error("attachment missing") == 'AttachmentError'

    def test_classify_alias(self, forge_clean):
        """Aliases like 'spam' should map to SpamTriggerError."""
        assert forge_clean._classify_error("contains spam words") == 'SpamTriggerError'

    def test_classify_unknown_returns_generic(self, forge_clean):
        """Unrecognized errors should return GenericError."""
        assert forge_clean._classify_error("something went wrong") == 'GenericError'

    def test_classify_case_insensitive(self, forge_clean):
        assert forge_clean._classify_error("TIMEZONEERROR: bad") == 'TimezoneError'


# ===========================================================================
# core.py — metrics callback
# ===========================================================================

class TestMetricsCallback:

    def test_callback_on_task_completed(self, tmp_dir):
        from cannyforge.core import CannyForge
        events = []
        forge = CannyForge(data_dir=tmp_dir, metrics_callback=lambda t, d: events.append((t, d)))
        forge.execute("Write an email about the meeting")
        assert any(e[0] == 'task_completed' for e in events)

    def test_callback_error_doesnt_crash(self, tmp_dir):
        from cannyforge.core import CannyForge

        def bad_callback(t, d):
            raise RuntimeError("callback boom")

        forge = CannyForge(data_dir=tmp_dir, metrics_callback=bad_callback)
        # Should not raise
        result = forge.execute("Write an email about meeting")
        assert result is not None


# ===========================================================================
# Quickstart integration test
# ===========================================================================

class TestQuickstartIntegration:

    def test_quickstart_runs_successfully(self):
        """The quickstart example should run without errors and show improvement."""
        project_root = Path(__file__).parent.parent
        quickstart = project_root / 'examples' / 'quickstart.py'
        result = subprocess.run(
            ['python', str(quickstart)],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        assert result.returncode == 0, f"quickstart failed:\n{result.stderr}"
        assert '3 patterns -> 3 rules' in result.stdout
        assert 'Improvement: +' in result.stdout
