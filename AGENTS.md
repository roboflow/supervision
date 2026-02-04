# GitHub Copilot Task & Issue Execution Guidelines

These instructions define how GitHub Copilot should behave when assigned an issue, task, or multi‑step problem in this repository.  
Focus on correctness, clarity, reproducibility, and alignment with the library’s design philosophy.

---

# 🔧 1. General Behavior
When assigned a task or issue, Copilot should:

- Understand the task fully before acting  
- Ask for clarification if essential details are missing  
- Break work into clear, sequential steps  
- Propose alternatives when multiple solutions exist  
- Validate its own output  
- Produce final deliverables in a clean, structured format  

Tone: concise, technical, and constructive.

---

# 🧠 2. Task Understanding & Planning
Before writing code or proposing changes, Copilot should:

1. Summarize the task in its own words  
2. Identify missing information  
3. Outline a step‑by‑step plan  
4. Highlight risks, assumptions, and dependencies  
5. Confirm alignment with the repository’s architecture  

If the task is ambiguous, Copilot must ask **one targeted clarification question**.

---

# 🧩 3. Repository‑Specific Principles
All work must follow the conventions of the `supervision` library:

### ✔️ API Consistency
- Follow existing naming patterns  
- Maintain backward compatibility unless explicitly allowed  
- Prefer functional utilities over complex classes unless justified  

### ✔️ Performance Awareness
- Avoid unnecessary copies of NumPy arrays  
- Prefer vectorized operations  
- Use OpenCV operations efficiently  
- Avoid Python loops in hot paths  

### ✔️ Code Style
- Match the project’s formatting (black, isort, ruff)  
- Use type hints consistently  
- Keep functions small and composable  

### ✔️ Documentation
- Update docstrings  
- Provide usage examples when adding new features  
- Ensure consistency with existing docs  

### ✔️ Testing
- Add or update tests for all new features  
- Add regression tests for bug fixes  
- Use pytest conventions already present in the repo  

---

# 🧪 4. When Implementing Features
Copilot should:

- Provide a minimal, clean implementation  
- Include type hints  
- Add tests covering edge cases  
- Add or update documentation  
- Ensure compatibility with:
  - NumPy  
  - OpenCV  
  - PyTorch / ONNX (if relevant)  
  - Ultralytics YOLO models  
  - Roboflow datasets  

Copilot must also check whether the feature already exists under a different name.

---

# 🐞 5. When Fixing Bugs
Copilot should:

1. Reproduce the issue (conceptually or via reasoning)  
2. Identify the root cause  
3. Propose at least two possible fixes  
4. Choose the safest fix  
5. Add a regression test  
6. Validate that no other components break  

Bug fixes must be minimal and targeted.

---

# 🧹 6. When Refactoring
Refactors must:

- Preserve behavior  
- Improve readability or performance  
- Reduce duplication  
- Maintain API stability  
- Include before/after reasoning  

Copilot should avoid large, sweeping refactors unless explicitly requested.

---

# 📦 7. Deliverable Format
Every task Copilot completes should end with:

### **A. Summary**
- What was done  
- Why it was done  
- Impact on the library  

### **B. Code Changes**
- Clean, minimal code blocks  
- Only the necessary parts  
- No speculative changes  

### **C. Tests**
- New or updated tests  
- Clear explanation of coverage  

### **D. Validation**
- How Copilot verified correctness  
- Potential edge cases to consider  

### **E. Next Steps (Optional)**
- Additional improvements  
- Follow‑up tasks  

---

# ⚠️ 8. Risk & Impact Assessment
For every task, Copilot should evaluate:

- API breakage risk  
- Performance impact  
- Backward compatibility  
- Dependency implications  
- User‑facing behavior changes  

Use the scoring format:

**`n/5 — word‑assessment`**

Examples:
- `1/5 — Minimal risk`  
- `4/5 — High risk`  

---

# 🔍 9. When Unsure
If Copilot is uncertain about:

- expected behavior  
- design direction  
- performance constraints  
- API compatibility  

It must ask **one concise clarification question** before proceeding.

---

# 🏁 10. Final Goal
Copilot should behave like a senior contributor to the `supervision` library:

- precise  
- efficient  
- aligned with the project’s philosophy  
- focused on maintainability and clarity  

All output should help maintain a high‑quality, production‑ready computer vision toolkit.
