from __future__ import annotations

from manga_rt_translator.models import Rect


def viewport_metrics_changed(
    previous: tuple[float, ...] | None,
    current: tuple[float, ...] | None,
) -> bool:
    """Detect resize, page zoom, or device-scale changes between polls."""
    return previous is not None and current is not None and current != previous


def scroll_position_script(
    region: Rect | None,
    viewport_width: int,
    viewport_height: int,
) -> str:
    if region is None or viewport_width <= 0 or viewport_height <= 0:
        x_ratio = y_ratio = 0.5
    else:
        x_ratio = min(1.0, max(0.0, (region.x + region.width / 2) / viewport_width))
        y_ratio = min(1.0, max(0.0, (region.y + region.height / 2) / viewport_height))
    x_value = format(x_ratio, ".8f")
    y_value = format(y_ratio, ".8f")
    return f"""(() => {{
  const x = Math.max(0, Math.min(window.innerWidth - 1, (window.innerWidth - 1) * {x_value}));
  const y = Math.max(0, Math.min(window.innerHeight - 1, (window.innerHeight - 1) * {y_value}));
  const target = document.elementFromPoint(x, y);
  let scroller = null;
  for (let node = target; node && node !== document.documentElement; node = node.parentElement) {{
    if (node === document.body) break;
    const style = window.getComputedStyle(node);
    const vertical = /^(auto|scroll|overlay)$/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 2;
    const horizontal = /^(auto|scroll|overlay)$/.test(style.overflowX) && node.scrollWidth > node.clientWidth + 2;
    if (vertical || horizontal) {{
      scroller = node;
      break;
    }}
  }}
  const metrics = {{width: window.innerWidth, height: window.innerHeight,
    dpr: window.devicePixelRatio || 1,
    documentWidth: document.documentElement.clientWidth,
    documentHeight: document.documentElement.clientHeight}};
  if (!scroller) {{
    const root = document.scrollingElement || document.documentElement;
    return JSON.stringify({{id: "window", x: Number(window.scrollX || root.scrollLeft || 0),
      y: Number(window.scrollY || root.scrollTop || 0), metrics}});
  }}
  const parts = [];
  for (let node = scroller; node && node !== document.documentElement; node = node.parentElement) {{
    let index = 0;
    for (let sibling = node.previousElementSibling; sibling; sibling = sibling.previousElementSibling) {{
      if (sibling.localName === node.localName) index++;
    }}
    parts.push(node.localName + ":" + index);
  }}
  return JSON.stringify({{id: parts.reverse().join("/"), x: Number(scroller.scrollLeft || 0),
    y: Number(scroller.scrollTop || 0), metrics}});
}})()"""


def scroll_requires_analysis(
    captured_y: float | None,
    current_y: float | None,
    moved_since_capture: bool,
) -> bool:
    return moved_since_capture or captured_y is None or current_y != captured_y


def plan_scroll_analysis(
    captured_y: float | None,
    current_y: float | None,
    moved_since_capture: bool,
    baseline_ready: bool,
    viewport_size: tuple[int, int] | None,
    capture_scale: float,
    *,
    captured_x: float | None = None,
    current_x: float | None = None,
    max_partial_fraction: float = 0.7,
) -> tuple[bool, Rect | None]:
    """Return whether to analyze and, when safe, the newly exposed strip."""
    moved = scroll_requires_analysis(captured_y, current_y, moved_since_capture)
    if captured_x is not None or current_x is not None:
        moved = moved or captured_x is None or current_x is None or current_x != captured_x
    if not moved:
        return False, None
    if not baseline_ready or viewport_size is None:
        return True, None
    width, height = viewport_size
    delta_y = (current_y - captured_y) if captured_y is not None and current_y is not None else 0
    delta_x = (current_x - captured_x) if captured_x is not None and current_x is not None else 0
    return True, exposed_scroll_region_2d(
        delta_x,
        delta_y,
        width,
        height,
        capture_scale,
        max_partial_fraction=max_partial_fraction,
    )


def promote_interrupted_scroll_to_full_view(
    analyze: bool,
    dirty_region: Rect | None,
    interrupted_work: bool,
) -> tuple[bool, Rect | None]:
    if interrupted_work:
        return True, None
    return analyze, dirty_region


def exposed_scroll_region(
    delta_css_pixels: float,
    width: int,
    height: int,
    capture_scale: float,
    *,
    max_partial_fraction: float = 0.7,
) -> Rect | None:
    """Return the newly exposed vertical strip, or None for a full reread."""
    return exposed_scroll_region_2d(
        0, delta_css_pixels, width, height, capture_scale,
        max_partial_fraction=max_partial_fraction,
    )


def exposed_scroll_region_2d(
    delta_x_css_pixels: float,
    delta_y_css_pixels: float,
    width: int,
    height: int,
    capture_scale: float,
    *,
    max_partial_fraction: float = 0.7,
) -> Rect | None:
    """Return a newly exposed single-axis strip; diagonal moves need a full read."""
    if width <= 0 or height <= 0 or capture_scale <= 0:
        return None
    if delta_x_css_pixels and delta_y_css_pixels:
        return None
    if delta_x_css_pixels:
        pixel_delta = int(round(abs(delta_x_css_pixels) * capture_scale))
        if pixel_delta <= 0 or pixel_delta >= width * max_partial_fraction:
            return None
        pixel_delta = min(pixel_delta, width)
        if delta_x_css_pixels > 0:
            return Rect(width - pixel_delta, 0, pixel_delta, height)
        return Rect(0, 0, pixel_delta, height)
    pixel_delta = int(round(abs(delta_y_css_pixels) * capture_scale))
    if pixel_delta <= 0 or pixel_delta >= height * max_partial_fraction:
        return None
    pixel_delta = min(pixel_delta, height)
    if delta_y_css_pixels > 0:
        return Rect(0, height - pixel_delta, width, pixel_delta)
    return Rect(0, 0, width, pixel_delta)
