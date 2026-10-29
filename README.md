# 📄 ATS Resume Checker

A Streamlit app that scores how ATS-friendly a resume is and suggests concrete improvements. Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An overall **ATS score** (0-100) and a breakdown by category
- Strengths, issues, and **missing keywords**
- Prioritised **improvement suggestions**
- Example **bullet rewrites**
- Quick rule-based checks (email, phone, sections, length)
- A downloadable JSON report

Built with [Streamlit](https://streamlit.io) and [Groq](https://groq.com) (default model: `llama-3.3-70b-versatile`).

## Project structure

```
.
├── app.py             # the Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free API key at <https://console.groq.com/keys>.
2. Install and run:

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"      # Windows PowerShell: $env:GROQ_API_KEY="your_key_here"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the sidebar of the running app.

## Push to GitHub (using the website, no command line)

1. Sign in at <https://github.com> and click **+** (top right) → **New repository**.
2. Name it (e.g. `ats-resume-checker`), choose **Public** (Streamlit Community Cloud's free tier works with public repos; private repos need access granted), and click **Create repository**.
3. On the new repo page, click **uploading an existing file** (or **Add file → Upload files**).
4. Drag and drop `app.py`, `requirements.txt` and `README.md` into the page.
5. Write a commit message such as `Initial commit` and click **Commit changes**.

> Never upload your API key. The app reads it from Streamlit secrets or an environment variable.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub (authorise access when asked).
2. Click **Create app** (or **New app**) → **Deploy a public app from GitHub**.
3. Pick your repository and branch (`main`), and set **Main file path** to `app.py`.
4. Click **Advanced settings** and, in the **Secrets** box, paste:

   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

5. Click **Deploy**. The first build takes a few minutes while dependencies install. You will get a public `*.streamlit.app` URL.

To change the key later: open your app → **⋮ / Manage app → Settings → Secrets**.

## Configuration

| Setting | Where | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Streamlit secrets or environment variable | Your Groq API key |
| `GROQ_MODEL` | Environment variable (optional) | Adds another model name to the sidebar dropdown |

The sidebar model list can be edited via `MODEL_CHOICES` in `app.py`. Groq's catalogue changes over time, so check <https://console.groq.com/docs/models> if a model stops working.

## Troubleshooting

- **"Very little text could be extracted"** – the resume is probably a scanned image. Export a text-based PDF or DOCX instead (an ATS can't read images either).
- **"This PDF is password protected"** – upload an unlocked copy.
- **Analysis failed / rate limit errors** – Groq's free tier has per-minute and per-day limits; wait a bit and retry.
- **Old `.doc` files** – not supported; save as `.docx` or PDF.

## Notes

- Scores come from an AI model, so treat them as guidance rather than an exact replica of any specific employer's ATS.
- Resume text is sent to Groq for analysis and is not stored by this app.
