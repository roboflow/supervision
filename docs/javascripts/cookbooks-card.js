// Event listener for when the DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {

    // Theme detection and setup
    const palette = __md_get("__palette");
    const useDark = palette && typeof palette.color === "object" && palette.color.scheme === "slate";
    const theme = useDark ? "dark-theme" : "light-default";

    // List of colors for labels to cycle through
    const colorList = [
        "#22c55e", "#14b8a6", "#ef4444", "#eab308",
        "#8b5cf6", "#f97316", "#3b82f6"
    ];

    // Fetch all repository cards in the document
    const repoCards = document.querySelectorAll(".repo-card");

    // Extract unique labels from each repo card, remove whitespace, and filter out empty labels
    const labelsAll = Array
        .from(repoCards)
        .flatMap((element) => element.getAttribute('data-labels').split(','))
        .map(label => label.trim())
        .filter(label => label !== '');
    const uniqueLabels = [...new Set(labelsAll)];

    // Map each unique label to a color from the color list
    const labelToColor = uniqueLabels.reduce((map, label, index) => {
        map[label] = colorList[index % colorList.length];
        return map;
    }, {});

    /**
     * Render card with repository details, including:
     * - Name, version, and authors
     * - Labels with dynamic background colors
     * - Author avatars with GitHub profile links
     * @param {HTMLElement} element - The repository card DOM element
     * @param {number} elementIndex - Index of the repository card element
     */
    async function renderCard(element, elementIndex) {
        const name = element.getAttribute('data-name');
        const labels = element.getAttribute('data-labels');
        const version = element.getAttribute('data-version');
        const authors = element.getAttribute('data-author');

        // Generate HTML for labels, assigning each a unique background color
        const labelHTML = labels ? labels.split(',')
            .filter(label => label !== '')
            .map((label) => {
                const color = labelToColor[label.trim()];
                return `
                    <span class="label non-selectable-text" style="background-color: ${color}">
                        ${label.trim()}
                    </span>
                `;
            }).join(' ') : '';

        // Fetch author information from GitHub API and render their avatar and name
        const authorArray = authors.split(',');
        const authorDataArray = await Promise.all(authorArray.map(async (author) => {
            const response = await fetch(`https://api.github.com/users/${author.trim()}`);
            return await response.json();
        }));

        // Generate HTML for author avatars, arranged with overlapping styles
        let authorAvatarsHTML = authorDataArray.map((authorData, index) => {
            const marginLeft = index === 0 ? '0' : '-10px';
            const zIndex = 4 - index;
            return `
                <div class="author-container" data-login="${authorData.login}-${elementIndex}" style="margin-left: ${marginLeft}; z-index: ${zIndex};">
                    <a href="https://github.com/${authorData.login}" target="_blank" style="line-height: 0;">
                        <img class="author-avatar" src="${authorData.avatar_url}" alt="${authorData.login}'s avatar">
                    </a>
                </div>
            `;
        }).join('');

        // Generate HTML for author names with links to GitHub profiles
        let authorNamesHTML = authorDataArray.map(
            authorData => `
                <span class="author-name" data-login="${authorData.login}-${elementIndex}">
                    <a href="https://github.com/${authorData.login}" target="_blank">
                        ${authorData.login}
                    </a>
                </span>
            `
        ).join(',&nbsp;');

        // Combine author avatar and name HTML into a single element
        let authorsHTML = `
            <div class="authors">
                ${authorAvatarsHTML}
                <div class="author-names">${authorNamesHTML}</div>
            </div>
        `;

        // Populate card HTML with name, version, authors, and labels
        element.innerText = `
            <div style="flex-direction: column; height: 100%; display: flex; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif; background: ${theme.background}; color: ${theme.color}; font-size: 14px; line-height: 1.5;">
                <div style="display: flex; align-items: center;">
                    <span style="font-weight: 700; font-size: 1rem;">
                        ${name}
                    </span>
                </div>
                ${authorsHTML}
                <div style="font-size: 12px; display: flex; justify-content: space-between;">
                    <div style="display: flex; align-items: center;">
                        <img src="/assets/supervision-lenny.png" aria-label="stars" width="20" height="20" role="img" />
                        <span style="margin-left: 4px">${version}</span>
                    </div>
                    <div style="display: flex; align-items: center; flex-wrap: wrap;">
                        ${labelHTML}
                    </div>
                </div>
            </div>
        `;

        // Sanitize the inner text and set as the element's HTML content
        let sanitizedHTML = DOMPurify.sanitize(element.innerText);
        element.innerHTML = sanitizedHTML;

        // Add event listeners for author name hover effects, toggling 'hover' class on avatar
        document.querySelectorAll('.author-name').forEach(element => {
            element.addEventListener('mouseenter', function () {
                const login = this.getAttribute('data-login');
                document.querySelector(`.author-container[data-login="${login}"]`).classList.add('hover');
            });

            element.addEventListener('mouseleave', function () {
                const login = this.getAttribute('data-login');
                document.querySelector(`.author-container[data-login="${login}"]`).classList.remove('hover');
            });
        });
    }

    // Render each repository card
    repoCards.forEach((element, index) => {
        renderCard(element, index);
    });
});
