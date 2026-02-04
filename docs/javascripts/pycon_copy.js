/**
 * Custom copy handler for Python console (pycon) code blocks.
 * Strips >>> and ... prompts when copying code examples.
 */
document.addEventListener("DOMContentLoaded", function () {
  const COPY_BUTTON_SELECTOR = ".md-clipboard, .md-code__button";

  function handleCopyButtonClick(event) {
    const copyButton = event.target.closest(COPY_BUTTON_SELECTOR);
    if (!copyButton) return;

    const codeBlock = findCodeBlockForCopyButton(copyButton);
    if (!codeBlock) return;

    const rawText = codeBlock.textContent || "";
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    const strippedText = stripPythonPrompts(rawText);
    primeClipboardButton(copyButton, strippedText);
  }

  function handleCopyButtonPointerDown(event) {
    const copyButton = event.target.closest(COPY_BUTTON_SELECTOR);
    if (!copyButton) return;

    const codeBlock = findCodeBlockForCopyButton(copyButton);
    if (!codeBlock) return;

    const rawText = codeBlock.textContent || "";
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    const strippedText = stripPythonPrompts(rawText);
    primeClipboardButton(copyButton, strippedText);
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
        button.removeEventListener(
          "pointerdown",
          handleCopyButtonPointerDown,
          true
        );
        button.addEventListener(
          "pointerdown",
          handleCopyButtonPointerDown,
          true
        );
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
  document.addEventListener("pointerdown", handleCopyButtonPointerDown, true);
  document.addEventListener("copy", handleSelectionCopy, true);
  bindCopyButtons(document);
  observeDynamicCopyButtons();
});

function primeClipboardButton(copyButton, strippedText) {
  copyButton.setAttribute("data-clipboard-text", strippedText);
  copyButton.removeAttribute("data-clipboard-target");
  copyButton.setAttribute("data-md-clipboard", "true");
}

function shouldStripPrompts(codeBlock, rawText) {
  const hasReplPrompts = /(^|\n)[ \t]*(>>>|\.\.\.)/.test(rawText);
  return (
    hasReplPrompts ||
    codeBlock.classList.contains("language-pycon") ||
    codeBlock.closest("pre")?.classList.contains("pycon") ||
    codeBlock.closest(".pycon") !== null ||
    codeBlock.closest(".highlight")?.classList.contains("pycon")
  );
}

function findCodeBlockForCopyButton(copyButton) {
  const targetSelector = copyButton.getAttribute("data-clipboard-target");
  if (targetSelector) {
    const target = document.querySelector(targetSelector);
    const targetCode = target?.querySelector?.("code") || target;
    if (targetCode?.tagName?.toLowerCase() === "code") {
      return targetCode;
    }
  }
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
 *
 * NOTE: This is a best-effort parser. It preserves unprompted lines inside
 * triple-quoted strings, but it does not fully model Python's tokenizer.
 */
function stripPythonPrompts(text) {
  const lines = text.split("\n");
  const codeLines = [];
  let inTripleQuotedString = false;

  function toggleTripleQuoteState(sourceLine) {
    const tripleQuotePattern = /("""|''')/g;
    const matches = sourceLine.match(tripleQuotePattern);
    if (!matches) return;
    if (matches.length % 2 === 1) {
      inTripleQuotedString = !inTripleQuotedString;
    }
  }

  for (const line of lines) {
    const trimmedLine = line.trimEnd();
    // Primary prompt: ">>> "
    if (trimmedLine.startsWith(">>> ")) {
      const stripped = trimmedLine.slice(4);
      codeLines.push(stripped);
      toggleTripleQuoteState(stripped);
    }
    // Continuation prompt: "... "
    else if (trimmedLine.startsWith("... ")) {
      const stripped = trimmedLine.slice(4);
      codeLines.push(stripped);
      toggleTripleQuoteState(stripped);
    }
    // Handle prompts without space after (edge case)
    else if (trimmedLine === ">>>") {
      codeLines.push("");
    }
    else if (trimmedLine === "...") {
      codeLines.push("");
    }
    else if (inTripleQuotedString) {
      codeLines.push(trimmedLine);
      toggleTripleQuoteState(trimmedLine);
    }
    // Skip output lines (lines that don't start with prompts)
    // This intentionally excludes output like "1.0" from the copied text
  }

  return codeLines.join("\n").trim();
}
