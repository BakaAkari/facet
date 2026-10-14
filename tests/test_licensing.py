"""Licence registry, policy semantics, generated docs and the doctor licence section."""

import importlib.util
import logging
import os
import re
import sqlite3
import subprocess
import sys

import pytest

import licensing
from db.schema import init_database
from diagnostics import run_doctor
from models.pyiqa_scorer import PYIQA_MODELS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _routing_names():
    """Loader names parsed from ModelManager.load_model_only (not hard-coded)."""
    with open(os.path.join(REPO_ROOT, 'models', 'model_manager.py'), encoding='utf-8') as f:
        src = f.read()
    start = src.index('loaders = {')
    block = src[start:src.index('\n        }', start)]
    loader_names = re.findall(r"^\s+'([\w+-]+)':", block, re.M)
    pyiqa_start = src.index('pyiqa_models = [', start)
    pyiqa_block = src[pyiqa_start:src.index(']', pyiqa_start)]
    return loader_names, re.findall(r"'([\w+-]+)'", pyiqa_block)


class TestRegistryCoverage:
    def test_every_pyiqa_model_is_registered(self):
        missing = [k for k in PYIQA_MODELS if k not in licensing.COMPONENTS]
        assert not missing

    def test_every_model_manager_loader_is_registered(self):
        loaders, pyiqa_names = _routing_names()
        assert len(loaders) >= 10 and len(pyiqa_names) >= 9  # the parse found the routing table
        missing = [n for n in loaders + pyiqa_names
                   if licensing.component_for_loader(n) not in licensing.COMPONENTS]
        assert not missing

    def test_required_ids_present(self):
        for cid in ('insightface', 'samp_net', 'u2netp', 'clip_mlp_head',
                    'mediapipe_face_landmarker', 'yunet', 'sface', 'ofiq', 'siglip2', 'clip',
                    'qwen3_5_2b', 'qwen3_5_4b', 'birefnet', 'deqa', 'aesthetic_v25', 'qrealign_native'):
            assert cid in licensing.COMPONENTS

    def test_registry_is_internally_consistent(self):
        for comp in licensing.COMPONENTS.values():
            assert comp.kind in ('code', 'weights', 'package')
            assert comp.commercial in ('yes', 'no', 'unclear')
            assert comp.replacement is None or comp.replacement in licensing.COMPONENTS


class TestPolicy:
    def test_default_mode_allows_everything(self):
        licensing.set_policy('default')
        assert all(licensing.allowed(cid) for cid in licensing.COMPONENTS)
        licensing.require('insightface')  # no raise

    def test_commercial_blocks_no_and_unclear(self):
        licensing.set_policy('commercial')
        assert licensing.allowed('sface')
        assert not licensing.allowed('insightface')
        assert not licensing.allowed('samp_net')  # unclear
        with pytest.raises(licensing.LicenceError, match='yunet'):
            licensing.require('insightface')
        with pytest.raises(licensing.LicenceError, match='allow_unclear'):
            licensing.require('samp_net')

    def test_allow_unclear_admits_unclear_but_not_no(self):
        licensing.set_policy('commercial', allow_unclear=True)
        assert licensing.allowed('samp_net')
        assert not licensing.allowed('topiq')

    def test_unknown_id_fails_loud(self):
        with pytest.raises(KeyError):
            licensing.allowed('nope')
        with pytest.raises(KeyError):
            licensing.require('nope')

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            licensing.set_policy('free')

    def test_lazy_resolution_from_config(self, monkeypatch):
        import config_resolve
        licensing.set_policy('default')
        licensing.reset_policy()
        monkeypatch.setattr(config_resolve, 'load_resolved',
                            lambda *a, **k: {'licensing': {'mode': 'commercial', 'allow_unclear': True}})
        assert licensing.policy() == licensing.Policy('commercial', True)

    def test_absent_block_is_default_mode(self):
        assert licensing.resolve_policy({}) == licensing.Policy('default', False)

    def test_unavailable_metrics(self, monkeypatch):
        licensing.set_policy('default')
        assert licensing.unavailable_metrics() == set()
        licensing.set_policy('commercial')
        real = importlib.util.find_spec
        monkeypatch.setattr(licensing.importlib.util, 'find_spec',
                            lambda name, *a: None if name == 'ofiq' else real(name, *a))
        assert licensing.unavailable_metrics() == {'liqe', 'aesthetic_iaa', 'face_quality_iqa'}
        monkeypatch.setattr(licensing.importlib.util, 'find_spec',
                            lambda name, *a: object() if name == 'ofiq' else real(name, *a))
        assert licensing.unavailable_metrics() == {'liqe', 'aesthetic_iaa'}


class TestRenderMarkdown:
    @pytest.mark.parametrize('lang', licensing.LANGUAGES)
    def test_one_row_per_component(self, lang):
        lines = licensing.render_markdown(lang).splitlines()
        assert len(lines) == len(licensing.COMPONENTS) + 2

    def test_unknown_language(self):
        with pytest.raises(KeyError):
            licensing.render_markdown('xx')


class TestGeneratedDocs:
    def test_docs_match_generator(self):
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'scripts', 'gen_third_party_models.py'), '--check'],
            capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize('lang', licensing.LANGUAGES)
    def test_markers_and_table_present(self, lang):
        parts = ['docs', 'THIRD_PARTY_MODELS.md'] if lang == 'en' else ['docs', lang, 'THIRD_PARTY_MODELS.md']
        with open(os.path.join(REPO_ROOT, *parts), encoding='utf-8') as f:
            text = f.read()
        assert '<!-- BEGIN GENERATED -->' in text and '<!-- END GENERATED -->' in text
        assert licensing.render_markdown(lang) in text


class _StdoutProxy:
    def write(self, msg):
        sys.stdout.write(msg)

    def flush(self):
        sys.stdout.flush()


class TestDoctorLicenceSection:
    @pytest.fixture(autouse=True)
    def _log_to_stdout(self):
        logger = logging.getLogger('facet.diagnostics')
        handler = logging.StreamHandler(_StdoutProxy())
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        yield
        logger.removeHandler(handler)

    def _doctor(self, capsys, tmp_path, config='{}', db=None):
        cfg = tmp_path / 'cfg.json'
        cfg.write_text(config, encoding='utf-8')
        run_doctor(config_path=str(cfg), db_path=str(db or tmp_path / 'none.db'))
        return capsys.readouterr().out

    def test_section_lists_mode_and_components(self, capsys, tmp_path):
        out = self._doctor(capsys, tmp_path)
        assert 'Model licences' in out
        assert 'Licensing mode: default' in out
        assert 'InsightFace buffalo_l (insightface)' in out

    def test_commercial_mode_lists_what_is_routed_not_what_is_configured(self, capsys, tmp_path):
        out = self._doctor(capsys, tmp_path, '{"licensing": {"mode": "commercial"}}')
        assert 'Licensing mode: commercial' in out
        # The scan loads the permissive stack ...
        assert 'YuNet (opencv_zoo) (yunet)' in out
        assert 'SFace (opencv_zoo) (sface)' in out
        # ... and never InsightFace, so the doctor does not warn about it.
        assert 'InsightFace buffalo_l (insightface)' not in out
        assert not re.search(r'\[!!\] .*not permitted in commercial mode', out)

    def test_face_model_mismatch_warning(self, capsys, tmp_path):
        db = tmp_path / 'mixed.db'
        init_database(str(db))
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO photos (path, filename) VALUES ('p/a.jpg', 'a.jpg')")
            conn.execute("INSERT INTO faces (photo_path, face_index, embedding, embedding_model) "
                         "VALUES ('p/a.jpg', 0, x'00', 'sface')")
            conn.execute("INSERT INTO faces (photo_path, face_index, embedding, embedding_model) "
                         "VALUES ('p/a.jpg', 1, x'00', 'arcface_buffalo_l')")
        out = self._doctor(capsys, tmp_path, db=db)
        assert "[!!] Face embedding model: 1 stored faces use 'sface'" in out

    def test_no_mismatch_warning_when_models_agree(self, capsys, tmp_path):
        db = tmp_path / 'clean.db'
        init_database(str(db))
        out = self._doctor(capsys, tmp_path, db=db)
        assert 'Face embedding model' not in out

    def test_missing_faces_table_is_graceful(self, capsys, tmp_path):
        db = tmp_path / 'bare.db'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE photos (path TEXT PRIMARY KEY)')
        out = self._doctor(capsys, tmp_path, db=db)
        assert 'Model licences' in out and 'Face embedding model' not in out


class TestLicensingModeEnv:
    """$FACET_LICENSING_MODE forces commercial over the file, on every resolution path."""

    def test_commercial_env_overrides_file(self, monkeypatch):
        monkeypatch.setenv('FACET_LICENSING_MODE', 'commercial')
        pol = licensing.resolve_policy({'licensing': {'mode': 'default'}})
        assert pol.mode == 'commercial'

    def test_lazy_policy_sees_env(self, monkeypatch):
        monkeypatch.setenv('FACET_LICENSING_MODE', 'commercial')
        licensing.reset_policy()
        assert licensing.policy().mode == 'commercial'

    @pytest.mark.parametrize('value', ['', 'default'])
    def test_default_or_empty_is_no_override(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_MODE', value)
        assert licensing.resolve_policy({'licensing': {'mode': 'commercial'}}).mode == 'commercial'
        assert licensing.resolve_policy({}).mode == 'default'

    @pytest.mark.parametrize('value', ['Commercial', ' COMMERCIAL '])
    def test_value_is_case_and_whitespace_insensitive(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_MODE', value)
        assert licensing.resolve_policy({}).mode == 'commercial'

    @pytest.mark.parametrize('value', ['bogus', 'comercial'])
    def test_invalid_value_raises(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_MODE', value)
        with pytest.raises(ValueError, match='FACET_LICENSING_MODE'):
            licensing.resolve_policy({})


class TestAllowUnclearIsABoolean:
    """06-1: a truthy non-bool must never grant the unclear-licence components."""

    @pytest.mark.parametrize('value', ['false', 'no', '0', 'true', '', 0, 1, 1.0, [], {}, None])
    def test_non_bool_in_config_raises(self, value):
        with pytest.raises(ValueError, match='allow_unclear'):
            licensing.resolve_policy({'licensing': {'mode': 'commercial', 'allow_unclear': value}})

    @pytest.mark.parametrize('value', ['false', 'no', 0, 1])
    def test_non_bool_raises_in_default_mode_too(self, value):
        with pytest.raises(ValueError, match='allow_unclear'):
            licensing.resolve_policy({'licensing': {'mode': 'default', 'allow_unclear': value}})

    @pytest.mark.parametrize('value', ['false', 'no', 0, 1])
    def test_set_policy_rejects_non_bool(self, value):
        with pytest.raises(ValueError, match='allow_unclear'):
            licensing.set_policy('commercial', value)

    @pytest.mark.parametrize('value', [True, False])
    def test_real_booleans_pass(self, value):
        pol = licensing.resolve_policy({'licensing': {'mode': 'commercial', 'allow_unclear': value}})
        assert pol.allow_unclear is value


class TestLicensingAllowUnclearEnv:
    """06-2: $FACET_LICENSING_ALLOW_UNCLEAR can only tighten, never grant."""

    CFG_TRUE = {'licensing': {'mode': 'commercial', 'allow_unclear': True}}
    CFG_FALSE = {'licensing': {'mode': 'commercial', 'allow_unclear': False}}

    @pytest.mark.parametrize('value', ['0', 'false', 'No', ' OFF '])
    def test_false_value_forces_unclear_off(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_ALLOW_UNCLEAR', value)
        assert licensing.resolve_policy(self.CFG_TRUE).allow_unclear is False

    @pytest.mark.parametrize('value', ['', '1', 'true', 'yes', 'on'])
    def test_true_or_empty_never_grants(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_ALLOW_UNCLEAR', value)
        assert licensing.resolve_policy(self.CFG_FALSE).allow_unclear is False
        assert licensing.resolve_policy(self.CFG_TRUE).allow_unclear is True

    @pytest.mark.parametrize('value', ['maybe', '2', 'flase'])
    def test_invalid_value_raises(self, monkeypatch, value):
        monkeypatch.setenv('FACET_LICENSING_ALLOW_UNCLEAR', value)
        with pytest.raises(ValueError, match='FACET_LICENSING_ALLOW_UNCLEAR'):
            licensing.resolve_policy(self.CFG_TRUE)

    def test_lazy_policy_sees_env(self, monkeypatch):
        import config_resolve
        monkeypatch.setattr(config_resolve, 'load_resolved', lambda *a, **k: self.CFG_TRUE)
        monkeypatch.setenv('FACET_LICENSING_ALLOW_UNCLEAR', '0')
        licensing.reset_policy()
        assert licensing.policy() == licensing.Policy('commercial', False)


class TestReraiseIfRefused:
    @staticmethod
    def _handled(exc_factory):
        """Run the helper inside a broad handler, the way every call site does."""
        try:
            try:
                raise exc_factory()
            except Exception:
                licensing.reraise_if_refused()
                return 'swallowed'
        except licensing.LicenceError:
            return 'refused'

    def test_a_direct_refusal_passes_through(self):
        assert self._handled(lambda: licensing.LicenceError('no')) == 'refused'

    def test_a_refusal_wrapped_with_from_passes_through(self):
        def make():
            try:
                raise licensing.LicenceError('no')
            except licensing.LicenceError as inner:
                err = RuntimeError('wrapped')
                err.__cause__ = inner
                return err
        assert self._handled(make) == 'refused'

    def test_a_refusal_in_the_context_chain_passes_through(self):
        def make():
            try:
                raise licensing.LicenceError('no')
            except licensing.LicenceError:
                try:
                    raise ValueError('during handling')
                except ValueError as e:
                    return e
        assert self._handled(make) == 'refused'

    def test_an_unrelated_error_is_left_for_the_handler(self):
        assert self._handled(lambda: ValueError('boom')) == 'swallowed'

    def test_a_context_cycle_terminates(self):
        a, b = ValueError('a'), ValueError('b')
        a.__context__, b.__context__ = b, a
        assert self._handled(lambda: a) == 'swallowed'

    def test_outside_an_except_block_it_does_nothing(self):
        licensing.reraise_if_refused()
