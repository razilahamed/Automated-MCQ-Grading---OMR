# OMR – NCEQ Assessment (V1/V2/V3)

This repo turns your Jupyter notebooks into a Python app with a simple UI (Streamlit or Gradio).

## Structure

```
omr_app/
├─ app_streamlit.py         # Streamlit UI
├─ app_gradio.py            # Gradio UI
├─ requirements.txt
├─ README.md
└─ omr_core/
   ├─ __init__.py
   ├─ utils.py              # shared helpers (image I/O, small utils)
   ├─ v1.py                 # paste functions from imageProcessV1.ipynb
   ├─ v2.py                 # paste functions from imageProcessV2.ipynb
   └─ v3.py                 # paste functions from imageProcessV3/withoutTemplateV3
```

## How to use

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Paste your logic**
   - Open `omr_core/v1.py`, `v2.py`, `v3.py`.
   - Replace the placeholder `process_v*` bodies with your real notebook functions (cleaned into pure Python).
   - Return this dict structure:
     ```python
     {
       "result": {
         "version": "vX",
         "score": int,
         "total": int,
         "decisions": [ ... ],   # any per-question details you want
         "meta": {...}
       },
       "debug_images": {
         "stage_name": np.ndarray (RGB),
         ...
       }
     }
     ```

3. **Run Streamlit UI**
   ```bash
   streamlit run app_streamlit.py
   ```

   Or **run Gradio UI**:
   ```bash
   python app_gradio.py
   ```

## Tips for migrating from notebooks

- Move helper cells into `utils.py`.
- Keep pure functions (no global state, no cv2.imshow/plt.show inside core functions).
- Make sure images returned in `debug_images` are **RGB** numpy arrays (H, W, 3) so the UIs can display them cleanly.
- If V3 is template-free, keep `template_bytes` but simply ignore it inside `process_v3`.

## Packaging (optional)

If you want to install as a package later:
```bash
pip install -e .
```
and create a minimal `pyproject.toml` with `project` metadata.


## Providing the answer key

You need an answer key to *grade* (compute correctness). Supply it via the UI in one of these formats:

- **Whitespace list**: `A B C D A C ...` → Q1='A', Q2='B', ...
- **CSV**:
  ```csv
  q,correct
  1,A
  2,B
  3,D
  ```
- **JSON** (supports multi-correct):
  ```json
  {"1":"A","2":["B","D"],"3":"C"}
  ```

The pipeline should return detected choices per question like:
```python
{
  "result": {
    "version": "vX",
    "detected": {
      "questions": {
        "1": {"chosen": ["A"], "conf": 0.95},
        "2": {"chosen": ["B","D"], "conf": 0.88}
      }
    }
  }
}
```
If your function instead returns `{"questions": ...}` at the top level, the UI will still find it.


### Using a scanned teacher **Key Sheet**
Both UIs can derive the answer key from a scanned key sheet:

- Upload the **Key Sheet** image (filled with correct bubbles).
- Enable **"Derive key from Key Sheet"** (Gradio) or select **"Derive from Key Sheet"** (Streamlit).
- The app runs the same detection pipeline on the key sheet and converts chosen bubbles to the answer key.

> Tip: Keep the key sheet printed on the same layout as student sheets to avoid alignment/offset issues.
