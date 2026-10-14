from manga_rt_translator.models import Rect
from manga_rt_translator.scroll_regions import (
    exposed_scroll_region,
    exposed_scroll_region_2d,
    plan_scroll_analysis,
    promote_interrupted_scroll_to_full_view,
    scroll_position_script,
    scroll_requires_analysis,
    viewport_metrics_changed,
)


def test_scroll_state_keeps_an_away_and_back_trip_actionable():
    assert scroll_requires_analysis(120.0, 120.0, moved_since_capture=True)
    assert not scroll_requires_analysis(120.0, 120.0, moved_since_capture=False)
    assert scroll_requires_analysis(120.0, 280.0, moved_since_capture=False)


def test_scroll_plan_skips_unchanged_view_and_reads_only_new_strip():
    assert plan_scroll_analysis(120, 120, False, True, (900, 600), 1.0) == (
        False,
        None,
    )
    assert plan_scroll_analysis(120, 200, True, True, (900, 600), 1.0) == (
        True,
        Rect(0, 520, 900, 80),
    )


def test_scroll_plan_falls_back_to_full_read_without_safe_baseline():
    assert plan_scroll_analysis(120, 200, True, False, (900, 600), 1.0) == (
        True,
        None,
    )
    assert plan_scroll_analysis(120, 120, True, True, (900, 600), 1.0) == (
        True,
        None,
    )


def test_interrupted_scroll_work_promotes_partial_analysis_to_full_view():
    assert promote_interrupted_scroll_to_full_view(
        True, Rect(0, 520, 900, 80), True
    ) == (True, None)
    assert promote_interrupted_scroll_to_full_view(
        True, Rect(0, 520, 900, 80), False
    ) == (True, Rect(0, 520, 900, 80))


def test_exposed_scroll_region_selects_the_new_bottom_strip_on_down_scroll():
    assert exposed_scroll_region(80, 900, 600, 1.0) == Rect(0, 520, 900, 80)


def test_exposed_scroll_region_selects_the_new_top_strip_on_up_scroll():
    assert exposed_scroll_region(-80, 900, 600, 1.5) == Rect(0, 0, 900, 120)


def test_scroll_position_script_finds_nested_scroller_without_mutating_page():
    script = scroll_position_script(Rect(20, 30, 100, 80), 200, 160)

    assert "document.elementFromPoint" in script
    assert "node.scrollHeight > node.clientHeight + 2" in script
    assert "node.scrollWidth > node.clientWidth + 2" in script
    assert 'JSON.stringify({id: parts.reverse().join("/")' in script
    assert "scrollTop || 0" in script
    assert "scrollLeft || 0" in script
    assert "devicePixelRatio" in script
    assert "setAttribute" not in script
    assert "dataset" not in script
    assert "0.35000000" in script
    assert "0.43750000" in script


def test_large_or_unusable_scroll_delta_requests_a_full_viewport_read():
    assert exposed_scroll_region(500, 900, 600, 1.0) is None
    assert exposed_scroll_region(0, 900, 600, 1.0) is None
    assert exposed_scroll_region(80, 0, 600, 1.0) is None


def test_horizontal_scroll_reads_only_the_newly_exposed_side():
    assert exposed_scroll_region_2d(80, 0, 900, 600, 1.0) == Rect(820, 0, 80, 600)
    assert exposed_scroll_region_2d(-80, 0, 900, 600, 1.5) == Rect(0, 0, 120, 600)


def test_diagonal_or_large_horizontal_scroll_requests_full_view():
    assert exposed_scroll_region_2d(80, 30, 900, 600, 1.0) is None
    assert exposed_scroll_region_2d(700, 0, 900, 600, 1.0) is None


def test_scroll_plan_tracks_horizontal_offsets_and_viewport_changes():
    assert plan_scroll_analysis(
        0, 0, False, True, (900, 600), 1.0, captured_x=10, current_x=90
    ) == (True, Rect(820, 0, 80, 600))
    assert plan_scroll_analysis(
        0, 0, True, True, (900, 600), 1.0, captured_x=None, current_x=90
    ) == (True, None)


def test_viewport_metrics_detect_zoom_dpi_and_resize_but_ignore_first_sample():
    baseline = (1200, 800, 1.0, 1200, 800)
    assert not viewport_metrics_changed(None, baseline)
    assert not viewport_metrics_changed(baseline, baseline)
    assert viewport_metrics_changed(baseline, (1200, 800, 1.25, 1200, 800))
    assert viewport_metrics_changed(baseline, (960, 640, 1.0, 960, 640))
