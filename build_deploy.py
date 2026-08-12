"""Build static deploy/ folder for Vercel from Flask/Jinja2 templates."""
import os, re, shutil

SRC = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(SRC, "deploy")

# Create folder structure
os.makedirs(os.path.join(DEPLOY, "css"), exist_ok=True)
os.makedirs(os.path.join(DEPLOY, "js"), exist_ok=True)

# Copy static assets
shutil.copy2(os.path.join(SRC, "frontend", "static", "css", "style.css"),
             os.path.join(DEPLOY, "css", "style.css"))
shutil.copy2(os.path.join(SRC, "frontend", "static", "js", "dashboard.js"),
             os.path.join(DEPLOY, "js", "dashboard.js"))
shutil.copy2(os.path.join(SRC, "frontend", "static", "js", "socket.io.min.js"),
             os.path.join(DEPLOY, "js", "socket.io.min.js"))

# Read sidebar partial
with open(os.path.join(SRC, "frontend", "templates", "partials", "sidebar.html"),
          "r", encoding="utf-8") as f:
    sidebar_raw = f.read()

# Pages to convert
pages = {
    "index.html": "dashboard",
    "analytics.html": "analytics",
    "students.html": "students",
    "alerts.html": "alerts",
    "settings.html": "settings",
}

for page, active in pages.items():
    with open(os.path.join(SRC, "frontend", "templates", page), "r", encoding="utf-8") as f:
        html = f.read()

    # Build sidebar with correct active class for this page
    sidebar = sidebar_raw
    for p_name in pages.values():
        pattern = "{{ 'active' if active_page == '" + p_name + "' }}"
        replacement = "active" if active == p_name else ""
        sidebar = sidebar.replace(pattern, replacement)

    # Fix sidebar links: / -> index.html, /analytics -> analytics.html, etc.
    sidebar = sidebar.replace('href="/"', 'href="index.html"')
    sidebar = sidebar.replace('href="/?autostart=1"', 'href="index.html"')
    sidebar = sidebar.replace('href="/analytics"', 'href="analytics.html"')
    sidebar = sidebar.replace('href="/students"', 'href="students.html"')
    sidebar = sidebar.replace('href="/alerts"', 'href="alerts.html"')
    sidebar = sidebar.replace('href="/settings"', 'href="settings.html"')

    # Replace Jinja {% include %} with inlined sidebar
    html = html.replace("{% include 'partials/sidebar.html' %}", sidebar)

    # Replace {{ url_for(...) }} with direct relative paths
    html = re.sub(
        r"\{\{\s*url_for\('static',\s*filename='css/style\.css'\)\s*\}\}\?v=\d+",
        "css/style.css", html
    )
    html = re.sub(
        r"\{\{\s*url_for\('static',\s*filename='js/dashboard\.js'\)\s*\}\}\?v=\d+",
        "js/dashboard.js", html
    )
    html = re.sub(
        r"\{\{\s*url_for\('static',\s*filename='js/socket\.io\.min\.js'\)\s*\}\}",
        "js/socket.io.min.js", html
    )

    # Write static HTML
    out_path = os.path.join(DEPLOY, page)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Created: deploy/{page}")

print("\nDone! deploy/ folder is ready for Vercel.")
print("Files:")
for root, dirs, files in os.walk(DEPLOY):
    for fname in files:
        rel = os.path.relpath(os.path.join(root, fname), DEPLOY)
        print(f"  deploy/{rel}")
