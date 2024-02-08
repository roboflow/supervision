
document.addEventListener("DOMContentLoaded", function () {

  async function setCard(el, url, name, desc, labels, version, theme, authors) {
    const colorList = [
      "A351FB", "FF4040", "FFA1A0", "FF7633", "FFB633", "D1D435", "4CFB12",
      "94CF1A", "40DE8A", "1B9640", "00D6C1", "2E9CAA", "00C4FF", "364797",
      "6675FF", "0019EF", "863AFF", "530087", "CD3AFF", "FF97CA", "FF39C9"
    ]

    let labelHTML = ''
    if (labels) {
      const labelArray = labels.split(',').map((label, index) => {
        const color = colorList[index % colorList.length]
        return `<span style="background-color: #${color}; color: #fff; padding: 2px 6px; border-radius: 12px; margin-right: 4px;">${label}</span>`
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
      const backgroundColor = theme === 'light-default' ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)';
      const textColor = theme === 'light-default' ? '#fff' : '#000';
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
      <div style="flex-direction: column; height: 100%; display: flex;
      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; background: ${theme.background}; font-size: 14px; line-height: 1.5; color: ${theme.color}">
        <div style="display: flex; align-items: center;">
          <i class="fa-solid:book-open" style="color: ${theme.color}; margin-right: 8px;"></i>
          <span style="font-weight: 600; color: ${theme.linkColor};">
            <a style="text-decoration: none; color: inherit;" href="${url}">${name}</a>
          </span>
        </div>
        <div style="font-size: 12px; margin-bottom: 10px; margin-top: 8px; color: ${theme.color}; flex: 1;">${desc}</div>
        <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 8px;">
            ${authorHTML}
        </div>
        <div style="font-size: 12px; color: ${theme.color}; display: flex; flex: 0;">
          <div style="display: 'flex'; align-items: center; margin-right: 16px;">
          </div>
          <div style="display: 'flex'; align-items: center; margin-right: 16px;">
            <img src="/assets/supervision-lenny.png" aria-label="stars" width="16" height="16" role="img" />
            &nbsp; <span>${version}</span>
          </div>
          <div style="display: 'flex'}; align-items: center;">
            &nbsp; <span>${labelHTML}</span>
          </div>
        </div>
      </div>
        `

    let sanitizedHTML = DOMPurify.sanitize(el.innerText);
    el.innerHTML = sanitizedHTML;
  }
  const themes = {
    'light-default': {
      background: 'white',
      borderColor: '#e1e4e8',
      color: '#586069',
      linkColor: '#0366d6',
    },
    'dark-theme': {
      background: '#1e2129',
      borderColor: '#607D8B',
      color: '#ECEFF1',
      linkColor: '#9E9E9E',
    }
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
