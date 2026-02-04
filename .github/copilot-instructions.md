# GitHub Copilot PR Review Guidelines

These rules ensure consistent, high‑signal, senior‑level reviews.  
Use emojis to improve navigation and clarity.

---

## 🔢 Global Scoring Rules
Whenever a score, rating, or qualitative label is used, follow this format:

**`n/N (optional:emoji) word‑assessment`**

Examples:
- `1/5 🔴 Critical`
- `3/5 🟡 Acceptable`
- `5/5 🟢 Excellent`

This applies to:
- Category scores  
- Risk levels  
- Any scale‑based evaluation  

---

## 🧭 1. Overall Recommendation
Choose one:
- 🟢 **Ready to Merge**
- 🟡 **Merge with Minor Fixes**
- 🟠 **Needs Significant Revisions**
- 🔴 **High Risk — Major Rework Required**
- ⛔ **Do Not Merge**

Add a one‑sentence justification.

---

## 📝 2. Executive Summary
3–5 bullets covering:
- What changed  
- Why it changed  
- Impact on behavior or architecture  
- Key risks or considerations  

---

## 📋 3. PR Template Compliance
Mark each as **Complete / Incomplete / Missing / N/A**:
- Problem statement  
- Description of changes  
- Screenshots/logs (if applicable)  
- Testing instructions  
- Breaking changes  
- Migration notes  
- Changelog entry  

---

## 🔍 4. Feature / Bug‑Fix Validation
Mark each as **Yes / Partially / No / N/A**.

### New Features:
- Tests added  
- Documentation updated  
- API changes documented  
- Backward compatibility considered  

### Bug Fixes:
- Regression test added  
- Root cause explained  
- Fix validated  
- No unintended side effects  

---

## 📊 5. Category Scores (1–5)
Provide a score + one‑sentence justification.

- **Code Correctness**  
- **Readability & Maintainability**  
- **Architecture & Design**  
- **Testing Quality**  
- **Documentation Quality**  
- **Risk Level** (1 = low, 5 = high)  

Use the global scoring rule: `n/5 — word‑assessment`.

---

## 🧠 6. Code & Design Review
Evaluate:
- Logic correctness, edge cases, error handling  
- Code smells or anti‑patterns  
- Modularity, API design, maintainability  
- Alignment with project conventions  

Include examples when useful.

---

## 🧪 7. Testing Review
Comment on:
- Coverage  
- Missing cases  
- Assertion quality  
- Real‑world scenario coverage  

---

## 💡 8. Suggestions for Improvement
Provide concise, actionable recommendations.  
Include code snippets when helpful.

---

## ⚠️ 9. Risk Assessment
Identify any:
- Breaking changes  
- Security issues  
- Performance concerns  
- Dependency risks  
- Migration implications  

---

## ✅ 10. Merge Checklist (Python Package)
Mark each as **Pass / Fail / N/A**.

### Static Analysis
- Linter passes  
- Type checks pass  
- Formatting matches standards  

### Tests
- All tests pass  
- New/updated tests included  
- Regression tests for fixes  

### Packaging
- Builds successfully (sdist + wheel)  
- Installs cleanly  
- Imports without errors  
- Version/changelog updated (if needed)  

### Documentation
- Docstrings updated  
- User-facing docs updated  

---

**Tone:** concise, objective, constructive.  
**Goal:** clear next actions for the PR author.
