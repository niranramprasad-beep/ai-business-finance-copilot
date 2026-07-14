# Vocab

Terms I've learned so far, in plain English. Updated as I go.

**terminal** — the app where I type commands for my Mac to run (the black window). Everything git and python related gets typed here, not on websites.

**command** — one instruction typed into the terminal, like `git push` or `python3 nike_challenge.py`. One command per line, Enter to run it.

**prompt (terminal)** — the line like `niranram@Nirans-MacBook-Pro ai-business-finance-copilot %` that means the terminal is ready for a new command. No prompt = something is still running.

**git** — the tool that tracks versions of my code. It remembers every snapshot I save so nothing is ever lost and I can see history.

**repository (repo)** — one project being tracked by git. My folder is a repo on my laptop, and there's a copy of it on GitHub.

**commit** — a saved snapshot of my code at one moment, with a message describing what changed. Lives on my laptop until I push.

**push** — uploads my commits from my laptop to GitHub. Nothing shows up on the GitHub website until I push.

**git add .** — stages all my changed files, which means "include these in the next commit." The dot means "everything in this folder."

**remote / origin** — the saved address of my repo's online copy. "origin" is just the default nickname for my GitHub repo's URL.

**token (GitHub)** — a special password GitHub gives me for terminal use, starts with ghp_. Real account passwords don't work in the terminal anymore.

**.gitignore** — a file listing what git should NEVER upload, like .env (my keys) and .venv (huge package folder).

**venv (virtual environment)** — a private copy of Python + packages just for this one project, in the .venv folder. Keeps projects from breaking each other. When my terminal shows (.venv), I'm inside it.

**pip** — Python's package installer. `pip install yfinance` downloads a library so my code can import it.

**library / package** — pre-written code I install and import instead of writing myself, like pandas or yfinance.

**requirements.txt** — a list of the packages my project needs, so anyone can install them all with one command.

**API** — a way for my code to talk to another company's service over the internet. yfinance uses Yahoo's, my LLM calls use Groq's and Google's.

**API key** — a secret string that proves my requests to an API are allowed, like a debit card number for the service. Anyone with the key can spend the account's quota, so it never goes in code, chat, or GitHub.

**.env** — the file where my secret keys live, one per line like `GROQ_API_KEY=gsk_...`. My code reads it with the dotenv library. It's in .gitignore so it never gets pushed.

**LLM (large language model)** — the AI that reads a prompt and writes text back, like the Groq llama model or Gemini. My copilot sends it data and it explains things in English.

**CSV** — a plain text spreadsheet file (comma separated values). superstore.csv is 10,000 rows of orders.

**DataFrame** — pandas' version of a spreadsheet inside Python: rows and columns I can filter, group, and do math on. yf.download and pd.read_csv both give me one.

**pandas** — the Python library for working with tables of data. Load a CSV, group by region, sum sales — all pandas.

**yfinance** — the library that downloads real stock market data from Yahoo Finance into a DataFrame.

**f-string** — a Python string with variables baked in, like `f"Highest close: ${price:.2f}"`. The :.2f part formats to 2 decimals.

**groupby** — the pandas move that splits rows into groups (like by Region) so I can total or average each group. `df.groupby("Region")["Sales"].sum()` = revenue per region.

**markdown (.md)** — a plain text format with light styling: # for headings, ** for bold, - for bullets. GitHub renders it as a nice document. This file is markdown.

**README.md** — the front-page file of a repo that explains what the project is. GitHub shows it automatically on the repo's main page.

**Jupyter notebook (.ipynb)** — code split into cells I can run one at a time, with outputs and charts saved right below each cell. Great for exploring data and presenting analysis; GitHub renders it with the charts visible.

**cell** — one block of a notebook. Run just that block without rerunning everything else.

**cURL** — a built-in terminal tool that sends a web request without writing any code. I used it to test my Gemini key.

**401 / 403 errors** — API rejections. 401 = "invalid credentials" (bad/dead key), 403 = "I know who you are but you're not allowed" (permissions problem).

**rate limit** — the cap on how many API requests I can send per minute/day on a free tier. Hitting it means wait, not pay.