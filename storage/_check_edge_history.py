import sqlite3, shutil, tempfile, os

src = r"C:\Users\20937\AppData\Local\Microsoft\Edge\User Data\Default\History"
tmp = os.path.join(tempfile.gettempdir(), "edge_history_copy.db")
shutil.copy2(src, tmp)
conn = sqlite3.connect(tmp)
rows = conn.execute(
    "select url, title from urls where url like '%github%' order by last_visit_time desc limit 30"
).fetchall()
if not rows:
    print("NO GITHUB HISTORY")
else:
    for url, title in rows:
        print(f"{url} | {title}")
