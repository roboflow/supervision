/*
 * Publishes the height of the version banner as a CSS variable.
 *
 * The banner is pinned to the top of the viewport, so the sticky header and
 * sidebars have to start below it instead of scrolling underneath. Their offset
 * cannot be hardcoded: the banner wraps to a different number of lines depending
 * on viewport width, and it stays hidden entirely on the latest release docs.
 */
(() => {
  const banner = document.querySelector("[data-md-component=outdated]");
  if (!banner) {
    return;
  }

  const publishHeight = () => {
    const height = banner.hidden ? 0 : banner.offsetHeight;
    document.documentElement.style.setProperty(
      "--sv-banner-height",
      `${height}px`,
    );
  };

  publishHeight();
  // Width changes reflow the text; Material flips `hidden` once its version
  // check decides the build is outdated, which is after this script runs.
  new ResizeObserver(publishHeight).observe(banner);
  new MutationObserver(publishHeight).observe(banner, {
    attributeFilter: ["hidden"],
    attributes: true,
  });
})();
