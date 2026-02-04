/**
 * Custom copy handler for Python console (pycon) code blocks.
 * Strips >>> and ... prompts when copying code examples.
 */
document.addEventListener("DOMContentLoaded", function () {
  const COPY_BUTTON_SELECTOR = ".md-clipboard";

  function handleCopyButtonClick(event) {
    const copyButton = event.target.closest(COPY_BUTTON_SELECTOR);
    if (!copyButton) return;

    const codeBlock = findCodeBlockForCopyButton(copyButton);
    if (!codeBlock) return;

    const rawText = codeBlock.textContent || "";
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    const strippedText = stripPythonPrompts(rawText);
    copyButton.setAttribute("data-clipboard-text", strippedText);
    copyButton.removeAttribute("data-clipboard-target");
  }

  function handleSelectionCopy(event) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    const anchorNode = range.commonAncestorContainer;
    const codeBlock =
      anchorNode.nodeType === Node.ELEMENT_NODE
        ? anchorNode.closest("code")
        : anchorNode.parentElement?.closest("code");

    if (!codeBlock) return;

    const rawText = selection.toString();
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    event.preventDefault();
    event.stopPropagation();

    const strippedText = stripPythonPrompts(rawText);
    event.clipboardData?.setData("text/plain", strippedText);
  }

  function bindCopyButtons(root) {
    root
      .querySelectorAll(COPY_BUTTON_SELECTOR)
      .forEach((button) => {
        button.removeEventListener("click", handleCopyButtonClick, true);
        button.addEventListener("click", handleCopyButtonClick, true);
      });
  }

  function observeDynamicCopyButtons() {
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type !== "childList") continue;
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType !== Node.ELEMENT_NODE) return;
          if (node.matches?.(COPY_BUTTON_SELECTOR)) {
            bindCopyButtons(node.parentElement || document);
            return;
          }
          if (node.querySelectorAll) {
            const hasButtons = node.querySelectorAll(COPY_BUTTON_SELECTOR);
            if (hasButtons.length > 0) {
              bindCopyButtons(node);
            }
          }
        });
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  document.addEventListener("click", handleCopyButtonClick, true);
  document.addEventListener("copy", handleSelectionCopy, true);
  bindCopyButtons(document);
  observeDynamicCopyButtons();
});

function shouldStripPrompts(codeBlock, rawText) {
  return (
    rawText.includes(">>>") ||
    rawText.includes("...") ||
    codeBlock.classList.contains("language-pycon") ||
    codeBlock.closest("pre")?.classList.contains("pycon") ||
    codeBlock.closest(".pycon") !== null ||
    codeBlock.closest(".highlight")?.classList.contains("pycon")
  );
}

function findCodeBlockForCopyButton(copyButton) {
  return (
    copyButton.closest("pre")?.querySelector("code") ||
    copyButton.parentElement?.querySelector("pre code") ||
    copyButton
      .closest(".highlight, .codehilite, .md-typeset__scrollwrap, .md-typeset")
      ?.querySelector("pre code") ||
    copyButton
      .closest(".highlight, .codehilite, .md-typeset__scrollwrap, .md-typeset")
      ?.querySelector("code")
  );
}

/**
 * Strips Python REPL prompts (>>> and ...) from code text.
 * Also removes output lines (lines that don't start with >>> or ...).
 */
function stripPythonPrompts(text) {
  const lines = text.split("\n");
  const codeLines = [];

  for (const line of lines) {
    const trimmedLine = line.trimEnd();
    // Primary prompt: ">>> "
    if (trimmedLine.startsWith(">>> ")) {
      codeLines.push(trimmedLine.slice(4));
    }
    // Continuation prompt: "... "
    else if (trimmedLine.startsWith("... ")) {
      codeLines.push(trimmedLine.slice(4));
    }
    // Handle prompts without space after (edge case)
    else if (trimmedLine === ">>>") {
      codeLines.push("");
    }
    else if (trimmedLine === "...") {
      codeLines.push("");
    }
    // Skip output lines (lines that don't start with prompts)
    // This intentionally excludes output like "1.0" from the copied text
  }

  return codeLines.join("\n").trim();
}

function copyText(text, copyButton) {
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(function () {
      showCopySuccess(copyButton);
    });
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
  showCopySuccess(copyButton);
}

function showCopySuccess(copyButton) {
  const originalIcon = copyButton.innerHTML;
  copyButton.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></svg>';
  setTimeout(function () {
    copyButton.innerHTML = originalIcon;
  }, 1500);
}
