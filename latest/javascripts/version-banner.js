/*
 * Publishes the height of the version banner as a CSS variable.
 *
 * The banner is pinned to the top of the viewport, so the sticky header and
 * sidebars have to start below it instead of scrolling underneath. Its height
 * cannot be hardcoded: the banner wraps at different viewport widths and stays
 * hidden entirely on the latest release docs.
 */
(() => {
  const banner = document.querySelector("[data-md-component=outdated]");
  if (!banner) {
    return;
  }

  const sidebarLayouts = new WeakMap();
  const sidebarBreakpoints = {
    navigation: window.matchMedia("(min-width: 76.25em)"),
    toc: window.matchMedia("(min-width: 60em)"),
  };

  const syncSidebar = (sidebar, bannerHeight) => {
    const scrollwrap = sidebar.querySelector(".md-sidebar__scrollwrap");
    const breakpoint = sidebarBreakpoints[sidebar.dataset.mdType];
    if (!scrollwrap || !breakpoint) {
      return;
    }

    const layout = sidebarLayouts.get(sidebar) ?? {
      adjustedHeight: "",
      adjustedTop: "",
      baseHeight: "",
      baseTop: "",
    };
    if (sidebar.style.top !== layout.adjustedTop) {
      layout.baseTop = sidebar.style.top;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      layout.baseHeight = scrollwrap.style.height;
    }

    if (!breakpoint.matches) {
      if (sidebar.style.top === layout.adjustedTop) {
        sidebar.style.top = layout.baseTop;
      }
      if (scrollwrap.style.height === layout.adjustedHeight) {
        scrollwrap.style.height = layout.baseHeight;
      }
      layout.adjustedTop = "";
      layout.adjustedHeight = "";
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    const baseTop = Number.parseFloat(layout.baseTop);
    const baseHeight = Number.parseFloat(layout.baseHeight);
    if (!Number.isFinite(baseTop) || !Number.isFinite(baseHeight)) {
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    layout.adjustedTop = `${baseTop + bannerHeight}px`;
    layout.adjustedHeight = `${baseHeight - bannerHeight}px`;
    if (sidebar.style.top !== layout.adjustedTop) {
      sidebar.style.top = layout.adjustedTop;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      scrollwrap.style.height = layout.adjustedHeight;
    }
    sidebarLayouts.set(sidebar, layout);
  };

  const syncSidebarLayout = () => {
    const bannerHeight = banner.hidden ? 0 : banner.offsetHeight;
    const sidebars = document.querySelectorAll("[data-md-component=sidebar]");
    for (const sidebar of sidebars) {
      syncSidebar(sidebar, bannerHeight);
    }
  };

  const publishHeight = () => {
    const height = banner.hidden ? 0 : banner.offsetHeight;
    document.documentElement.style.setProperty(
      "--sv-banner-height",
      `${height}px`,
    );
    syncSidebarLayout();
  };

  publishHeight();
  // Width changes reflow the text; Material flips `hidden` once its version
  // check decides the build is outdated, which is after this script runs.
  new ResizeObserver(publishHeight).observe(banner);
  new MutationObserver(publishHeight).observe(banner, {
    attributeFilter: ["hidden"],
    attributes: true,
  });
  const sidebarObserver = new MutationObserver(syncSidebarLayout);
  for (const sidebar of document.querySelectorAll("[data-md-component=sidebar]")) {
    sidebarObserver.observe(sidebar, {
      attributeFilter: ["style"],
      attributes: true,
      subtree: true,
    });
  }
  for (const breakpoint of Object.values(sidebarBreakpoints)) {
    breakpoint.addEventListener("change", syncSidebarLayout);
  }
})();
