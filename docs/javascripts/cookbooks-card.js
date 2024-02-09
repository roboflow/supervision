
document.addEventListener("DOMContentLoaded", function () {

  async function setCard(el, url, name, labels, version, theme, authors) {
    const colorList = [
      "#22c55e",
      "#14b8a6",
      "#ef4444",
      "#eab308",
      "#8b5cf6",
      "#f97316",
      "#3b82f6",
    ]

    let labelHTML = ''
    if (labels) {
      const labelArray = labels.split(',').map((label, index) => {
        const color = colorList[index % colorList.length]
        return `<span class="non-selectable-text" style="background-color: ${color}; color: #fff; padding: 2px 4px; border-radius: 4px; margin-right: 4px;">${label}</span>`
      })

      labelHTML = labelArray.join(' ')
    }

    const authorArray = authors.split(',');
    const authorDataArray = await Promise.all(authorArray.map(async (author) => {
      const response = await fetch(`https://api.github.com/users/${author.trim()}`);
      return await response.json();
    }));

    let authorAvatarsHTML = authorDataArray.map((authorData, index) => {
        const marginLeft = index === 0 ? '0' : '-10px';
        const zIndex = 100 - index;
        return `
            <div class="author-container" style="margin-left: ${marginLeft}; z-index: ${zIndex};">
                <a href="https://github.com/${authorData.login}" target="_blank" style="line-height: 0;">
                    <img class="author-avatar" src="${authorData.avatar_url}" alt="${authorData.login}'s avatar">
                </a>
            </div>
        `;
    }).join('');

    let authorNamesHTML = authorDataArray.map(
        authorData => `<span class="author-name" style="color: ${theme.color}">
            <a href="https://github.com/${authorData.login}" target="_blank">
                ${authorData.login}
            </a>
        </span>`
    ).join(',&nbsp;');

    let authorsHTML = `
        <div class="authors">
            ${authorAvatarsHTML}
            <div class="author-names">${authorNamesHTML}</div>
        </div>
    `;

    el.innerText = `
      <a style="text-decoration: none; color: inherit;" href="${url}">
        <div style="flex-direction: column; height: 100%; display: flex;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; background: ${theme.background}; font-size: 14px; line-height: 1.5; color: ${theme.color}">
          <div style="display: flex; align-items: center;">
            <span style="font-weight: 700; font-size: 1rem; color: ${theme.linkColor};">
              ${name}
            </span>
          </div>
          ${authorsHTML}
          <div style="font-size: 12px; color: ${theme.color}; display: flex; flex: 0; justify-content: space-between">
            <div style="display: flex; align-items: center;">
              <img src="/assets/supervision-lenny.png" aria-label="stars" width="20" height="20" role="img" />
              &nbsp;
              <span style="margin-left: 4px">${version}</span>
            </div>
            <div style="display: flex; align-items: center;">
              ${labelHTML}
            </div>
          </div>
        </div>
      </a>
        `

    let sanitizedHTML = DOMPurify.sanitize(el.innerText);
    el.innerHTML = sanitizedHTML;
  }
  for (const el of document.querySelectorAll('.repo-card')) {
    const url = el.getAttribute('data-url');
    const name = el.getAttribute('data-name');
    const labels = el.getAttribute('data-labels');
    const version = el.getAttribute('data-version');
    const authors = el.getAttribute('data-author');
    const palette = __md_get("__palette")
    if (palette && typeof palette.color === "object") {
      var theme = palette.color.scheme === "slate" ? "dark-theme" : "light-default"
    } else {
      var theme = "light-default"
    }

    setCard(el, url, name, labels, version, theme, authors);
  }
})
