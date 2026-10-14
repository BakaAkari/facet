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

    def test_commercial_mode_warns_for_restricted_components(self, capsys, tmp_path):
        out = self._doctor(capsys, tmp_path, '{"licensing": {"mode": "commercial"}}')
        assert 'Licensing mode: commercial' in out
        assert re.search(r'\[!!\] InsightFace buffalo_l \(insightface\): not permitted', out)

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
