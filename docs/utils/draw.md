---
comments: true
---

# Draw Utils

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_line">draw_line</a></h2>
</div>

:::supervision.draw.utils.draw_line

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_rectangle">draw_rectangle</a></h2>
</div>

:::supervision.draw.utils.draw_rectangle

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_filled_rectangle">draw_filled_rectangle</a></h2>
</div>

:::supervision.draw.utils.draw_filled_rectangle

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_polygon">draw_polygon</a></h2>
</div>

:::supervision.draw.utils.draw_polygon

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_filled_polygon">draw_filled_polygon</a></h2>
</div>

:::supervision.draw.utils.draw_filled_polygon

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_text">draw_text</a></h2>
</div>

:::supervision.draw.utils.draw_text

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.draw_image">draw_image</a></h2>
</div>

:::supervision.draw.utils.draw_image

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.calculate_optimal_text_scale">calculate_optimal_text_scale</a></h2>
</div>

:::supervision.draw.utils.calculate_optimal_text_scale

<div class="md-typeset">
    <h2><a href="#supervision.draw.utils.calculate_optimal_line_thickness">calculate_optimal_line_thickness</a></h2>
</div>

:::supervision.draw.utils.calculate_optimal_line_thickness

<div class="md-typeset">
    <h2><a href="#supervision.draw.color.Color">Color</a></h2>
</div>

:::supervision.draw.color.Color

**New in 0.24.0:**
- Supports 4- and 8-digit hex codes (e.g., `#f0f8`, `#ff00ff80`) for RGBA colors.
- The `Color` class now has an `a` (alpha) field (default 255, fully opaque).
- `as_hex()` returns `#RRGGBBAA` if alpha is not 255, otherwise `#RRGGBB`.
- Use `as_rgba()` to get an (r, g, b, a) tuple.

**Examples:**
```python
import supervision as sv

sv.Color.from_hex('#ff00ff80')  # Color(r=255, g=0, b=255, a=128)
sv.Color.from_hex('#f0f8')      # Color(r=255, g=0, b=255, a=136)
sv.Color(r=255, g=0, b=255, a=128).as_hex()  # '#ff00ff80'
sv.Color(r=255, g=0, b=255, a=128).as_rgba() # (255, 0, 255, 128)
```

<div class="md-typeset">
    <h2><a href="#supervision.draw.color.ColorPalette">ColorPalette</a></h2>
</div>

:::supervision.draw.color.ColorPalette
