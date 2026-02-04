# GitHub Copilot PR Review Guidelines

These guidelines ensure consistent, actionable, and maintainer-focused reviews for the Supervision project.

---

## 🎯 Review Objectives

Your primary goal is to help maintainers make informed merge decisions quickly by:

1. **Validating PR completeness** against project requirements
2. **Identifying risks** that could impact users or maintainability
3. **Providing actionable feedback** the author can immediately act upon
4. **Highlighting quality gaps** in code, tests, or documentation

---

## 🟢 1. Overall Recommendation

**Start with a clear, actionable recommendation:**

Choose one and provide a **specific** justification:

- 🟢 **Approve** — Ready to merge as-is
- 🟡 **Minor Suggestions** — Minor improvements recommended but not blocking
- 🟠 **Request Changes** — Significant issues must be addressed before merge
- 🔴 **Block** — Critical issues require major rework

**Example:**

```
🟠 Request Changes — Missing unit tests for new `PolygonMerger` class and no documentation entry added for autogeneration.
```

---

## 📋 2. PR Completeness Check

Verify the PR meets project requirements. Mark each item:

- ✅ **Complete** — Properly addressed
- ⚠️ **Incomplete** — Partially done, needs improvement
- ❌ **Missing** — Not provided
- 🔵 **N/A** — Not applicable to this PR

### Required Items

- [ ] **Clear description** — What changed and why
- [ ] **Type of change** — Bug fix, feature, docs, etc.
- [ ] **Motivation/context** — Problem being solved (links to issue if relevant)
- [ ] **Changes list** — Summary of modifications
- [ ] **Tests** — Unit tests added/updated
- [ ] **Documentation** — Docstrings follow [Google-style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [ ] **Docs entry** — Added to mkdocs for autogeneration (new functions/classes only)
- [ ] **Google Colab** — Provided for demonstrating feature/fix (if applicable)
- [ ] **Screenshots/videos** — Included for visual changes (if applicable)

**Call out missing items explicitly:**

```
❌ Missing:
- Documentation entry not added to mkdocs navigation
- No unit tests provided for `merge_polygons()` function
```

---

## 📊 3. Quality Assessment

### 3.1 Code Quality

Provide **specific feedback using inline comments** on the changed code. Use **n/5** scoring for quick assessment:

- **5/5** 🟢 Excellent — Well-structured, idiomatic, no issues
- **4/5** 🟢 Good — Minor improvements possible
- **3/5** 🟡 Acceptable — Some issues to address
- **2/5** 🟠 Needs Work — Multiple problems
- **1/5** 🔴 Poor — Significant refactoring required

**Score: n/5** — [Brief justification]

#### Check for:

1. **Correctness**

    - Logic errors or edge cases not handled
    - Potential bugs (None checks, array bounds, division by zero)
    - Incorrect assumptions

2. **Python Best Practices**

    - Non-idiomatic patterns
    - Improper exception handling
    - Inefficient implementations
    - Missing or incorrect type hints

3. **Project Conventions**

    - **Docstrings:** Must follow [Google-style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
    - **Code style:** Must pass linting (run `pre-commit run --all-files`)
    - **Imports:** Standard library → third-party → local
    - **Naming:** Clear, descriptive, follows PEP 8

**Place inline comments directly on problematic code**, then reference them in your summary. Example:

```markdown
See inline comments in `detection/core.py` for:
- Null safety issue in mask processing (see inline comment on mask handling)
- Performance concern with nested loops (see inline comment on nested loops)
```

### 3.2 Testing Quality

Use **n/5** scoring for test coverage and quality:

- **5/5** 🟢 Comprehensive — All cases covered, high-quality assertions
- **4/5** 🟢 Good — Most cases covered
- **3/5** 🟡 Adequate — Basic coverage, some gaps
- **2/5** 🟠 Insufficient — Major gaps
- **1/5** 🔴 Missing — No tests or tests don't validate functionality

**Score: n/5** — [Brief justification]

#### For New Features or Bug Fixes:

1. **Coverage Requirements**

    - [ ] Unit tests added for new functions/classes
    - [ ] Edge cases covered (empty inputs, None, large arrays, boundary conditions)
    - [ ] Regression tests for bug fixes

2. **Test Quality**

    - [ ] Assertions are specific (not just "no exception raised")
    - [ ] Tests use realistic scenarios
    - [ ] Test names clearly describe what they validate

**If tests are inadequate, comment on test files and reference:**

```
2/5 🟠 Insufficient Testing - See comments in `test/detection/test_zone.py`
```

### 3.3 Documentation Quality

Use **n/5** scoring for documentation completeness:

- **5/5** 🟢 Excellent — Complete, clear, with good examples
- **4/5** 🟢 Good — Minor improvements possible
- **3/5** 🟡 Adequate — Basic docs present
- **2/5** 🟠 Insufficient — Incomplete or unclear
- **1/5** 🔴 Missing — No documentation

**Score: n/5** — [Brief justification]

#### For New Features:

1. **Docstring Requirements**

    - [ ] Docstrings for all public functions/classes
    - [ ] Parameters, return values, and exceptions documented
    - [ ] Usage examples in docstrings

2. **Documentation Integration**

    - [ ] Entry added to appropriate docs page (e.g., `docs/detection/tools/*.md`)
    - [ ] Added to mkdocs navigation (`mkdocs.yml`)
    - [ ] Changelog entry (`docs/changelog.md`) for user-facing changes

#### For Changes to Existing Features:

1. **Update Requirements**
    - [ ] Docstrings updated to reflect changes
    - [ ] Deprecated features marked with warnings
    - [ ] Migration guide for breaking changes

**Comment on docstrings directly in code, then reference in summary.**

---

## ⚠️ 4. Risk Assessment

**Explicitly flag any risks with severity:**

- **5/5** 🔴 Critical — Blocks release, must fix
- **4/5** 🟠 High — Serious concern, should fix
- **3/5** 🟡 Medium — Notable risk, consider fixing
- **2/5** 🟢 Low — Minor concern
- **1/5** 🟢 Negligible — No real risk

### Risk Categories:

1. **Breaking Changes**

    - Changes to public APIs (function signatures, return types)
    - Removal of deprecated features
    - Changed behavior in existing functionality
    - **If breaking:** Must include migration instructions

2. **Performance Impact**

    - Inefficient algorithms ($O(n^2)$ where $O(n)$ possible)
    - Memory-intensive operations on large arrays
    - Potential bottlenecks in hot paths

3. **Compatibility Issues**

    - New Python version requirements
    - New dependencies
    - Platform-specific code

4. **Security Concerns**

    - Unvalidated user input
    - Potential code execution risks
    - Sensitive data exposure

**Example:**

```
Risk Level: 4/5 🟠 High Performance Risk

Nested loop detected - see inline comment in `zone.py` for vectorization suggestion.
```

---

## 💡 5. Constructive Suggestions

**Add inline comments to the code using GitHub's review interface**, then provide **suggested changes** using GitHub suggestion format:

````markdown
```suggestion
if detections is None or detections.mask is None:
    return None
return process(detections.mask)
```
````

### Suggestion Categories:

1. **Code Improvements**

    - Logic simplifications
    - Better error handling
    - More readable implementations

2. **Performance Optimizations**

    - NumPy vectorization opportunities
    - Caching expensive computations
    - Batch processing

3. **Architecture Improvements**

    - Code reuse opportunities
    - Better abstractions
    - More maintainable designs

**Best Practice:** Place suggestions as inline comments on specific code blocks, then summarize in your review.

---

## 📊 6. Review Summary Template

Use this structure for your final review comment:

```markdown
## Review Summary

### Recommendation
[emoji] [Status] — [One-sentence justification]

### PR Completeness
- ✅ Complete: [list key items]
- ❌ Missing: [list critical gaps]

### Quality Scores
- **Code Quality:** n/5 [emoji] — [brief reason]
- **Testing:** n/5 [emoji] — [brief reason]
- **Documentation:** n/5 [emoji] — [brief reason]

### Risk Level: n/5 [emoji]
[Brief risk description with reference to inline comments if applicable]

### Critical Issues (Must Fix)
1. [Issue description] — See comment on `file.py`
2. [Another blocking issue] — See comment on `test_file.py`

### Suggestions (Nice to Have)
1. [Improvement idea] — See suggestion on `file.py`
2. [Another optional enhancement]

### Next Steps for Author
1. [Clear action item]
2. [Another clear action item]
```

---

## 🎯 Best Practices for Effective Reviews

### DO:

1. ✅ **Use n/5 scoring** for quick assessment of quality dimensions
2. ✅ **Place comments directly on code** using GitHub's inline comment feature
3. ✅ **Use GitHub suggestion format** for code changes when possible
4. ✅ **Reference inline comments** in your summary (e.g., "See comment on `file.py:function()`")
5. ✅ **Explain *why*** something is a problem, not just *what* is wrong
6. ✅ **Distinguish** between blocking issues and nice-to-haves
7. ✅ **Acknowledge** good work and clever solutions
8. ✅ **Run linter** locally if needed: `uv run pre-commit run --all-files`

### DON'T:

1. ❌ **Don't mention line numbers** in summary — place comments inline instead
2. ❌ **Don't give vague feedback** like "improve code quality"
3. ❌ **Don't nitpick** on personal style preferences (defer to automated tools)
4. ❌ **Don't assume** the author knows project conventions
5. ❌ **Don't focus only on problems** — recognize what's good
6. ❌ **Don't let perfect** be the enemy of good (minor issues shouldn't block useful PRs)

---

## 📝 Workflow

1. **Review files** in the PR, placing inline comments on specific issues
2. **Use GitHub suggestions** for concrete code improvements
3. **Draft your summary** using the template above
4. **Reference inline comments** instead of mentioning specific line numbers
5. **Submit review** with clear recommendation and next steps

---

## 🗣️ Tone and Communication

- **Be respectful and constructive** — Contributors are volunteers
- **Be specific and technical** — Help them learn
- **Be pragmatic** — Balance ideal vs. practical
- **Be consistent** — Follow these guidelines every time

**Remember:** Your goal is to help maintainers efficiently assess PRs and help contributors improve their work. Focus on **actionable feedback** that moves the PR toward merge.
