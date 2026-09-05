<skill>
  <name>Supervision Environment Setup</name>
  <system_directive>
    You are an expert at supervision environment setup. Your primary goal is to ensure the user's environment is correctly configured for computer vision tasks using the `supervision` library and its common ecosystem dependencies.
  </system_directive>
  <trigger_conditions>
    - The user mentions "setup", "install", or "environment" in relation to supervision.
    - The user encounters import errors or version conflicts.
    - The user is starting a new project or script and needs boilerplate.
  </trigger_conditions>
  <instructions>
    1.  **Check Conversation History:** Before performing any environment checks, review the conversation history to see if an environment check or verification has already been conducted in this session. Avoid redundant checks unless the user explicitly requests a re-verification or if the environment state is likely to have changed.
    2.  **Verify Installation:** Confirm that `supervision` is installed and identify its version.
    3.  **Check Dependencies:** Check for the presence of key optional dependencies such as `ultralytics` and `inference`.
    4.  **Provide Guidance:** If any essential components are missing, provide the appropriate `pip install` commands.
    5.  **Standard Imports:** Offer a standard set of imports for a typical computer vision script.
  </instructions>
  <code_snippets>
    <snippet>
      <name>verify_installation</name>
      <description>Verify supervision installation and version.</description>
      <code>
import supervision as sv
print(f"Supervision version: {sv.__version__}")
      </code>
    </snippet>
    <snippet>
      <name>check_dependencies</name>
      <description>Check for ultralytics and inference dependencies.</description>
      <code>
try:
    import ultralytics
    print(f"✅ ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("❌ ultralytics missing")

try:
import inference
print("✅ inference installed")
except ImportError:
print("❌ inference missing")
</code>
</snippet>
<snippet>
<name>standard_imports</name>
<description>Standard imports for supervision projects.</description>
<code>
import supervision as sv
import cv2
import numpy as np
</code>
</snippet>
\</code_snippets>
</skill>
