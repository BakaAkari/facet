"""RAM-cache policy by available bytes (scan-pass performance spec, step S6a).

``ModelManager`` admits a model to the CPU RAM cache only while the host keeps
``model + ram_cache_reserve_gb`` free, and ``evict_cpu_cache`` can free memory
down to a target of available bytes, one model at a time in Belady order,
instead of always wiping the whole cache.

Nothing here loads a model. Memory readings come through the manager's
``memory_provider`` seam, so no global is patched; the fake provider reports
``base_available + the size of every model that left the cache``, which is what
a real eviction does to the host.
"""

from __future__ import annotations

import logging
import types
from unittest import mock

import pytest

from utils import system_memory
from utils.system_memory import EffectiveMemory

GIB = 1024 ** 3


def _fake_torch_module():
    fake = types.SimpleNamespace()
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=0),
        memory_allocated=lambda: 0, memory_reserved=lambda: 0,
    )
    return fake


def _make_config(keep_in_ram='auto', **extra):
    model_config = {
        'vram_profile': 'legacy',
        'keep_in_ram': keep_in_ram,
        'profiles': {'legacy': {'aesthetic_model': 'clip-mlp',
                                'composition_model': 'rule-based'}},
        **extra,
    }

    class _Cfg:
        def get_model_config(self):
            return dict(model_config)

    return _Cfg()


@pytest.fixture(autouse=True)
def _torch_free():
    fake = _fake_torch_module()
    with mock.patch('models.model_manager._ensure_torch', return_value=fake), \
         mock.patch('models.model_manager.torch', fake), \
         mock.patch('utils.device.get_device', return_value='cpu'):
        yield


def _available(n_bytes):
    return lambda: EffectiveMemory(64 * GIB, 64 * GIB - n_bytes, n_bytes, 50.0)


def _manager(available_bytes=None, provider=None, **config):
    from models.model_manager import ModelManager
    if provider is None and available_bytes is not None:
        provider = _available(available_bytes)
    return ModelManager(_make_config(**config), memory_provider=provider)


class _Scorer:
    """A cached wrapper that counts ``report_skips`` calls (torch-free)."""

    def __init__(self):
        self.reports = 0

    def report_skips(self):
        self.reports += 1


def _cached_manager(names=('a', 'b', 'c'), size=GIB, base=0, call_cap=200):
    """A manager holding ``names`` in its RAM cache and a provider that reports
    ``base`` plus ``size`` for every model that is no longer cached."""
    holder = {}
    calls = {'n': 0}

    def provider():
        calls['n'] += 1
        assert calls['n'] <= call_cap, 'the eviction loop does not terminate'
        mgr = holder['manager']
        freed = sum(size for n in names if n not in mgr._cpu_cache)
        return EffectiveMemory(64 * GIB, 0, base + freed, 50.0)

    manager = _manager(provider=provider)
    holder['manager'] = manager
    manager._cpu_cache = {n: _Scorer() for n in names}
    manager.provider_calls = calls
    return manager


class TestAdmissionReserve:
    """``_can_cache_to_ram`` keeps today's form ``available_gb > model + reserve``."""

    def test_the_reserve_is_public_and_defaults_to_four_gb(self):
        assert _manager().ram_cache_reserve_gb == 4.0

    def test_available_equal_to_model_plus_reserve_is_refused(self):
        topiq = 2.0  # MODEL_RAM_REQUIREMENTS['topiq']
        assert not _manager(int((topiq + 4.0) * GIB))._can_cache_to_ram('topiq')

    def test_one_byte_more_is_admitted(self):
        topiq = 2.0
        assert _manager(int((topiq + 4.0) * GIB) + 1)._can_cache_to_ram('topiq')

    def test_a_configured_reserve_admits_where_the_default_refuses(self):
        available = int(4.0 * GIB)  # topiq 2.0 + default 4.0 = 6.0 needed
        assert not _manager(available)._can_cache_to_ram('topiq')
        assert _manager(available, ram_cache_reserve_gb=1.0)._can_cache_to_ram('topiq')

    def test_a_larger_reserve_refuses_where_the_default_admits(self):
        available = int(7.0 * GIB)
        assert _manager(available)._can_cache_to_ram('topiq')
        assert not _manager(available, ram_cache_reserve_gb=6.0)._can_cache_to_ram('topiq')

    def test_a_zero_reserve_is_valid(self):
        assert _manager(int(2.5 * GIB), ram_cache_reserve_gb=0)._can_cache_to_ram('topiq')

    def test_never_never_reads_memory(self):
        provider = mock.Mock(side_effect=AssertionError('read memory'))
        assert not _manager(provider=provider, keep_in_ram='never')._can_cache_to_ram('clip')
        provider.assert_not_called()

    def test_always_never_reads_memory_and_keeps_the_membership_gate(self):
        provider = mock.Mock(side_effect=AssertionError('read memory'))
        m = _manager(provider=provider, keep_in_ram='always')
        assert m._can_cache_to_ram('clip')
        assert not m._can_cache_to_ram('vlm_tagger')
        provider.assert_not_called()

    def test_always_does_not_need_the_reserve_or_provider_attributes(self):
        """The licensing gate calls ``_can_cache_to_ram`` on a bare stub."""
        from models.model_manager import ModelManager
        stub = types.SimpleNamespace(keep_in_ram='always', device='cpu', _cpu_plan=None,
                                     CPU_CACHEABLE_MODELS=ModelManager.CPU_CACHEABLE_MODELS)
        assert ModelManager._can_cache_to_ram(stub, 'clip')

    @pytest.mark.parametrize('bad', ['abc', -0.5, -1, None, True, float('nan'), float('inf'), [4]])
    def test_an_invalid_reserve_raises_naming_the_key(self, bad):
        with pytest.raises(ValueError, match='ram_cache_reserve_gb'):
            _manager(ram_cache_reserve_gb=bad)

    def test_a_numeric_reserve_is_stored_as_a_float(self):
        m = _manager(ram_cache_reserve_gb=2)
        assert m.ram_cache_reserve_gb == 2.0 and isinstance(m.ram_cache_reserve_gb, float)


class TestMemoryProviderSeam:
    def test_an_injected_provider_is_used_for_available_bytes(self):
        assert _manager(123456)._available_bytes() == 123456

    def test_the_default_provider_resolves_effective_memory_at_call_time(self, monkeypatch):
        m = _manager()
        monkeypatch.setattr(system_memory, 'effective_memory', _available(777))
        assert m._available_bytes() == 777
        monkeypatch.setattr(system_memory, 'effective_memory', _available(888))
        assert m._available_bytes() == 888

    def test_admission_reads_through_the_seam_not_the_global(self, monkeypatch):
        monkeypatch.setattr(system_memory, 'effective_memory',
                            mock.Mock(side_effect=AssertionError('used the global')))
        assert _manager(int(10 * GIB))._can_cache_to_ram('topiq')


class TestEvictCpuCacheWithoutTarget:
    def test_evicts_everything_and_returns_the_names(self):
        m = _cached_manager()
        assert m.evict_cpu_cache() == ['a', 'b', 'c']
        assert m._cpu_cache == {}

    def test_logs_the_count_and_names_as_before(self, caplog):
        m = _cached_manager()
        with caplog.at_level(logging.INFO, logger='facet.models'):
            m.evict_cpu_cache()
        assert 'Evicted 3 model(s) from RAM cache: a, b, c' in caplog.text

    def test_never_consults_memory_without_a_target(self):
        m = _cached_manager()
        m.evict_cpu_cache()
        assert m.provider_calls['n'] == 0

    def test_an_empty_cache_returns_an_empty_list_and_logs_nothing(self, caplog):
        m = _cached_manager(names=())
        with caplog.at_level(logging.INFO, logger='facet.models'):
            assert m.evict_cpu_cache() == []
        assert 'Evicted' not in caplog.text

    def test_reports_each_scorer_once(self):
        m = _cached_manager()
        scorers = list(m._cpu_cache.values())
        m.evict_cpu_cache()
        assert [s.reports for s in scorers] == [1, 1, 1]

    def test_releases_the_heap_once(self, monkeypatch):
        release = mock.Mock(return_value=True)
        monkeypatch.setattr(system_memory, 'release_freed_heap', release)
        _cached_manager().evict_cpu_cache()
        release.assert_called_once_with()


class TestEvictCpuCacheWithTarget:
    def test_stops_as_soon_as_the_target_is_met(self):
        m = _cached_manager(base=0)
        evicted = m.evict_cpu_cache(needed_available_bytes=GIB)
        assert evicted == ['a']
        assert list(m._cpu_cache) == ['b', 'c']

    def test_evicts_only_as_many_as_needed(self):
        m = _cached_manager(base=0)
        assert m.evict_cpu_cache(needed_available_bytes=2 * GIB) == ['a', 'b']
        assert list(m._cpu_cache) == ['c']

    def test_a_target_already_met_evicts_nothing(self):
        m = _cached_manager(base=5 * GIB)
        assert m.evict_cpu_cache(needed_available_bytes=GIB) == []
        assert list(m._cpu_cache) == ['a', 'b', 'c']

    def test_a_target_equal_to_the_reading_is_met(self):
        m = _cached_manager(base=GIB)
        assert m.evict_cpu_cache(needed_available_bytes=GIB) == []

    def test_one_byte_short_evicts_one_model(self):
        m = _cached_manager(base=GIB - 1)
        assert m.evict_cpu_cache(needed_available_bytes=GIB) == ['a']

    def test_belady_evicts_the_farthest_next_use_first(self):
        m = _cached_manager(base=0)
        assert m.evict_cpu_cache(needed_available_bytes=GIB, next_use=['c', 'b', 'a']) == ['a']
        assert list(m._cpu_cache) == ['b', 'c']

    def test_belady_order_for_several_victims(self):
        m = _cached_manager(base=0)
        assert m.evict_cpu_cache(needed_available_bytes=2 * GIB,
                                 next_use=['b', 'c', 'a']) == ['a', 'c']

    def test_names_missing_from_next_use_go_first(self):
        m = _cached_manager(base=0)
        assert m.evict_cpu_cache(needed_available_bytes=GIB, next_use=['a', 'b']) == ['c']

    def test_insertion_order_breaks_ties_among_unlisted_names(self):
        m = _cached_manager(names=('d', 'c', 'b', 'a'), base=0)
        assert m.evict_cpu_cache(needed_available_bytes=2 * GIB, next_use=['a']) == ['d', 'c']

    def test_no_next_use_falls_back_to_insertion_order(self):
        m = _cached_manager(names=('z', 'y', 'x'), base=0)
        assert m.evict_cpu_cache(needed_available_bytes=2 * GIB) == ['z', 'y']

    def test_next_use_may_be_any_iterable_and_may_repeat_names(self):
        m = _cached_manager(base=0)
        order = iter(['a', 'b', 'c', 'a'])  # 'a' is needed first; later repeats do not matter
        assert m.evict_cpu_cache(needed_available_bytes=GIB, next_use=order) == ['c']

    def test_an_unreachable_target_evicts_everything_and_terminates(self):
        m = _cached_manager(base=0)
        evicted = m.evict_cpu_cache(needed_available_bytes=100 * GIB)
        assert evicted == ['a', 'b', 'c']
        assert m._cpu_cache == {}
        assert m._available_bytes() < 100 * GIB  # the caller can tell emptied from reached

    def test_an_unreachable_target_on_a_provider_that_never_moves(self):
        reads = []

        def stuck():
            reads.append(1)
            assert len(reads) < 100, 'the eviction loop does not terminate'
            return EffectiveMemory(64 * GIB, 64 * GIB, 0, 100.0)

        m = _manager(provider=stuck)
        m._cpu_cache = {'a': _Scorer(), 'b': _Scorer(), 'c': _Scorer()}
        assert m.evict_cpu_cache(needed_available_bytes=GIB) == ['a', 'b', 'c']

    def test_an_empty_cache_with_a_target_returns_an_empty_list(self):
        m = _cached_manager(names=(), base=0)
        assert m.evict_cpu_cache(needed_available_bytes=GIB) == []

    def test_each_victim_is_reported_and_the_heap_released_per_victim(self, monkeypatch):
        release = mock.Mock(return_value=True)
        monkeypatch.setattr(system_memory, 'release_freed_heap', release)
        m = _cached_manager(base=0)
        scorers = dict(m._cpu_cache)
        m.evict_cpu_cache(needed_available_bytes=2 * GIB)
        assert [scorers[n].reports for n in 'abc'] == [1, 1, 0]
        assert release.call_count == 2

    def test_logs_the_eviction_at_info_and_the_target_at_debug(self, caplog):
        m = _cached_manager(base=0)
        with caplog.at_level(logging.DEBUG, logger='facet.models'):
            m.evict_cpu_cache(needed_available_bytes=GIB)
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any('Evicted 1 model(s) from RAM cache: a' in r.getMessage() for r in info)
        assert any(str(GIB) in r.getMessage() for r in debug)

    def test_a_target_that_is_already_met_logs_no_eviction(self, caplog):
        m = _cached_manager(base=5 * GIB)
        with caplog.at_level(logging.INFO, logger='facet.models'):
            m.evict_cpu_cache(needed_available_bytes=GIB)
        assert 'Evicted' not in caplog.text

    def test_resident_models_are_never_touched(self):
        m = _cached_manager(base=0)
        resident = object()
        m.models['clip'] = resident
        m.evict_cpu_cache(needed_available_bytes=100 * GIB)
        assert m.models == {'clip': resident}


class TestEvictionSurvivesAConcurrentRestore:
    """The targeted loop keeps the single-``pop`` discipline (R17): a model the
    scan thread restored between the snapshot and the pop is skipped, never
    raised on and never reported as evicted."""

    class _RacingCache(dict):
        def pop(self, key, *default):
            if self:
                dict.pop(self, list(self)[-1])  # the scan thread restores the newest
            return super().pop(key, *default)

    def test_a_restore_between_the_snapshot_and_the_pop_is_skipped(self):
        m = _cached_manager(base=0)
        m._cpu_cache = self._RacingCache({'a': _Scorer(), 'b': _Scorer(), 'c': _Scorer()})
        evicted = m.evict_cpu_cache(needed_available_bytes=100 * GIB)
        assert evicted == ['a']          # only what this call really took
        assert dict(m._cpu_cache) == {}  # b and c went to the scan thread

    def test_a_restore_of_the_next_victim_does_not_stop_the_loop(self):
        m = _cached_manager(base=0)
        m._cpu_cache = {'a': _Scorer(), 'b': _Scorer(), 'c': _Scorer()}
        original = m._memory_provider

        def restore_b_then_read():
            m._cpu_cache.pop('b', None)
            return original()

        m._memory_provider = restore_b_then_read
        evicted = m.evict_cpu_cache(needed_available_bytes=100 * GIB)
        assert 'b' not in evicted
        assert m._cpu_cache == {}

    def test_the_count_logged_is_what_the_call_really_evicted(self, caplog):
        m = _cached_manager(base=0)
        m._cpu_cache = self._RacingCache({'a': _Scorer(), 'b': _Scorer()})
        with caplog.at_level(logging.INFO, logger='facet.models'):
            m.evict_cpu_cache(needed_available_bytes=100 * GIB)
        assert 'Evicted 2 model(s)' not in caplog.text


class TestIsResidentOrCached:
    def test_true_for_a_resident_model(self):
        m = _manager()
        m.models['clip'] = object()
        assert m.is_resident_or_cached('clip')

    def test_true_for_a_cached_model(self):
        m = _manager()
        m._cpu_cache['topiq'] = object()
        assert m.is_resident_or_cached('topiq')

    def test_false_otherwise(self):
        m = _manager()
        m.models['clip'] = object()
        m._cpu_cache['topiq'] = object()
        assert not m.is_resident_or_cached('liqe')

    def test_has_no_side_effects(self):
        m = _manager()
        cached = object()
        m._cpu_cache['topiq'] = cached
        hits, misses = m._cache_hits, m._cache_misses
        for name in ('topiq', 'clip', 'liqe'):
            m.is_resident_or_cached(name)
        assert m._cpu_cache == {'topiq': cached} and m.models == {}
        assert (m._cache_hits, m._cache_misses) == (hits, misses)
