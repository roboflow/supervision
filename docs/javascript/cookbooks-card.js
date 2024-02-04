window.addEventListener('DOMContentLoaded', async function () {

    function setCard(el, name,desc,labels,version, theme) {

        el.innerHTML = `
        <div style="flex-direction: column; height: 100%; display: flex;
          font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; background: ${theme.background}; font-size: 14px; line-height: 1.5; color: #24292e;">
          <div style="display: flex; align-items: center;">
            <i class="fa-solid:book-open" style="color: ${theme.color}; margin-right: 8px;"></i>
            <span style="font-weight: 600; color: ${theme.linkColor};">
              <a style="text-decoration: none; color: inherit;" href="${name}">${name}</a>
            </span>
          </div>
          <div style="font-size: 12px; margin-bottom: 16px; margin-top: 8px; color: ${theme.color}; flex: 1;">${desc}</div>
          <div style="font-size: 12px; color: ${theme.color}; display: flex; flex: 0;">
            <div style="display: 'flex'; align-items: center; margin-right: 16px;">
            </div>
            <div style="display: 'flex'; align-items: center; margin-right: 16px;">
            <img src="/assets/supervision-lenny.png" aria-label="stars" width="16" height="16" role="img" />
            &nbsp; <span>${version}</span>
          </div>
            <div style="display: 'flex'}; align-items: center;">
              <svg style="fill: ${theme.color};" aria-label="fork" viewBox="0 0 16 16" version="1.1" width="16" height="16" role="img"><path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path></svg>
              &nbsp; <span>${labels}</span>
            </div>
          </div>
        </div>
        `;
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
    };

    for (const el of document.querySelectorAll('.repo-card')) {
        const name = el.getAttribute('data-name');
        const desc = el.getAttribute('data-desc');
        const labels = el.getAttribute('data-labels');
        const version = el.getAttribute('data-version');
        const theme = themes[el.getAttribute('data-theme') || 'dark-theme'];
        console.log(name, desc, labels, version, theme); // Check if the attributes are being read correctly

        setCard(el, name, desc, labels, version, theme);
    }


});


// listen to hash change
window.onhashchange = function() {
    // get the new hash
    var newHash = window.location.hash.substring(1);
    // check if the new hash is the same as the old hash
    if (newHash === oldHash) {
        // if it is, it means the user clicked the back button
        // so we can call the function that we want
        // in this case, we want to close the modal
        alert("Back button clicked");
    } else {
        // if the new hash is different from the old hash
        // it means the user clicked the back button
        // so we can call the function that we want
        // in this case, we want to open the modal
        alert("Forward button clicked");
    }
}
