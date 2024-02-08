
document.addEventListener("DOMContentLoaded", function () {

  async function setCard(el, url, name, desc, labels, version, theme, authors) {
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

    let authorHTML = '';
    authorDataArray.forEach((authorData, index) => {
      const marginLeft = index === 0 ? '0' : '-15px';
      authorHTML += `
          <div class="author-container" style="display: inline-block; margin-left: ${marginLeft}; position: relative;">
          <a href="https://github.com/${authorData.login}" target="_blank">
            <img src="${authorData.avatar_url}" width="32" height="32" style="border-radius: 50%;" />
          </a>
            <div class="tooltip" style="visibility: hidden; background-color: #555; color: #fff; text-align: center; border-radius: 6px; padding: 5px 0; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -60px; opacity: 0; transition: opacity 0.3s; width: 120px;">
            ${authorData.login}
          </div>
          </div>
        `;
    });

    document.querySelectorAll('.author-container').forEach((container) => {
      const tooltip = container.querySelector('.tooltip');
      container.addEventListener('mouseover', () => {
        tooltip.style.visibility = 'visible';
        tooltip.style.opacity = '1';
      });
      container.addEventListener('mouseout', () => {
        tooltip.style.visibility = 'hidden';
        tooltip.style.opacity = '0';
      });
    });



    el.innerText = `
      <a style="text-decoration: none; color: inherit;" href="${url}">
        <div style="flex-direction: column; height: 100%; display: flex;
        font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; background: ${theme.background}; font-size: 14px; line-height: 1.5; color: ${theme.color}">
          <div style="display: flex; align-items: center;">
            <span style="font-weight: 700; font-size: 1rem; color: ${theme.linkColor};">
              ${name}
            </span>
          </div>
          <div style="font-size: 12px; margin-top: 0.5rem; color: ${theme.color}; flex: 1;">
            ${desc}
          </div>
          <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 1rem; margin-top: 1rem">
              ${authorHTML}
          </div>
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
    const desc = el.getAttribute('data-desc');
    const labels = el.getAttribute('data-labels');
    const version = el.getAttribute('data-version');
    const authors = el.getAttribute('data-author');
    const palette = __md_get("__palette")
    if (palette && typeof palette.color === "object") {
      var theme = palette.color.scheme === "slate" ? "dark-theme" : "light-default"
    } else {
      var theme = "light-default"
    }

    setCard(el, url, name, desc, labels, version, theme, authors);
  }
})
