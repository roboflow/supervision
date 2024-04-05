document.addEventListener("DOMContentLoaded", function () {

    const palette = __md_get("__palette")
    const useDark = palette && typeof palette.color === "object" && palette.color.scheme === "slate"
    const theme = useDark ? "dark-theme" : "light-default";

    const colorList = [
        "#22c55e",
        "#14b8a6",
        "#ef4444",
        "#eab308",
        "#8b5cf6",
        "#f97316",
        "#3b82f6",
    ]

    const repoCards = document.querySelectorAll(".repo-card");
    const labelsAll = Array
        .from(repoCards)
        .flatMap((element) => element.getAttribute('data-labels').split(','))
        .map(label => label.trim())
        .filter(label => label !== '');
    const uniqueLabels = [...new Set(labelsAll)];

    const labelToColor = uniqueLabels.reduce((map, label, index) => {
        map[label] = colorList[index % colorList.length];
        return map;
    }, {});


  async function renderCard(element, elementIndex) {
    const url = element.getAttribute('data-url');
    const name = element.getAttribute('data-name');
    const labels = element.getAttribute('data-labels');
    const version = element.getAttribute('data-version');
    const authors = element.getAttribute('data-author');

    const labelHTML = labels ? labels.split(',').filter(label => label !== '').map((label, index) => {
        const color = labelToColor[label.trim()];
        return `
            <span
                class="label non-selectable-text"
                style="background-color: ${color}"
            >
                ${label.trim()}
            </span>
        `;
    }).join(' ') : '';

    const authorArray = authors.split(',');
    const authorDataArray = await Promise.all(authorArray.map(async (author) => {
      const response = await fetch(`https://api.github.com/users/${author.trim()}`);
      return await response.json();
    }));

    let authorAvatarsHTML = authorDataArray.map((authorData, index) => {
        const marginLeft = index === 0 ? '0' : '-10px';
        const zIndex = 4 - index;
        return `
            <div
                class="author-container"
                data-login="${authorData.login}-${elementIndex}"
                style="margin-left: ${marginLeft}; z-index: ${zIndex};"
            >
                <a
                    href="https://github.com/${authorData.login}"
                    target="_blank"
                    style="line-height: 0;"
                >
                    <img
                        class="author-avatar"
                        src="${authorData.avatar_url}"
                        alt="${authorData.login}'s avatar"
                    >
                </a>
            </div>
        `;
    }).join('');

    let authorNamesHTML = authorDataArray.map(
        authorData => `
            <span
                class="author-name"
                data-login="${authorData.login}-${elementIndex}"
                style="color: ${theme.color}"
            >
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

    element.innerText = `
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
            <div style="display: flex; align-items: center; flex-wrap: wrap">
              ${labelHTML}
            </div>
          </div>
        </div>
      </a>
        `

    let sanitizedHTML = DOMPurify.sanitize(element.innerText);
    element.innerHTML = sanitizedHTML;

    document.querySelectorAll('.author-name').forEach(element => {
        element.addEventListener('mouseenter', function() {
            const login = this.getAttribute('data-login');
            document.querySelector(`.author-container[data-login="${login}"]`).classList.add('hover');
        });

        element.addEventListener('mouseleave', function() {
            const login = this.getAttribute('data-login');
            document.querySelector(`.author-container[data-login="${login}"]`).classList.remove('hover');
        });
    });
  }
    repoCards.forEach((element, index) => {
        renderCard(element, index);
    });
})
